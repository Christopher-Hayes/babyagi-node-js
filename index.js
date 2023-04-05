#!/usr/bin/env node
/* eslint-disable no-console */
/* eslint-disable no-await-in-loop */
const openai = require("openai");
const pinecone = require("pinecone-node-sdk");
const dotenv = require("dotenv");

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || "us-east1-gcp";
const YOUR_TABLE_NAME = process.env.TABLE_NAME;
const OBJECTIVE = process.argv[2] || process.env.OBJECTIVE;
const YOUR_FIRST_TASK = process.env.FIRST_TASK;
const USE_GPT4 = false;

if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY environment variable is missing from .env");
}
if (!PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY environment variable is missing from .env");
}
if (!YOUR_TABLE_NAME) {
  throw new Error("TABLE_NAME environment variable is missing from .env");
}
if (!OBJECTIVE) {
  throw new Error("OBJECTIVE environment variable is missing from .env");
}
if (!YOUR_FIRST_TASK) {
  throw new Error("FIRST_TASK environment variable is missing from .env");
}

console.log("\x1b[96m\x1b[1m\n*****OBJECTIVE*****\n\x1b[0m\x1b[0m");
console.log(OBJECTIVE);

openai.apiKey = OPENAI_API_KEY;
pinecone.initializeApp(PINECONE_API_KEY, PINECONE_ENVIRONMENT);

const tableName = YOUR_TABLE_NAME;
const dimension = 1536;
const metric = "cosine";
const podType = "p1";
pinecone.listIndexes().then((indexes) => {
  if (!indexes.includes(tableName)) {
    pinecone.createIndex(tableName, dimension, metric, podType);
  }
});

const index = new pinecone.Index(tableName);
const taskList = [];

function addTask(task) {
  taskList.push(task);
}

function get_ada_embedding(text) {
  text = text.replace("\n", " ");
  return openai.Embedding.create({
    input: [text],
    model: "text-embedding-ada-002",
  }).then((response) => response.data[0].embedding);
}

async function openAICall(
  prompt,
  useGpt4 = false,
  temperature = 0.5,
  max_tokens = 100
) {
  if (!useGpt4) {
    const response = await openai.Completion.create({
      engine: "text-davinci-003",
      prompt,
      temperature,
      max_tokens,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
    });
    return response.choices[0].text.trim();
  }
  const response = await openai.ChatCompletion.create({
    model: "gpt-4",
    messages: [{ role: "user", content: prompt }],
    temperature,
    max_tokens,
    n: 1,
    stop: null,
  });
  return response.choices[0].message.content.trim();
}

async function taskCreationAgent(
  objective,
  result,
  taskDescription,
  taskList,
  gptVersion = "gpt-3"
) {
  const prompt = `You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: ${objective}, The last completed task has the result: ${result}. This result was based on this task description: ${taskDescription}. These are incomplete tasks: ${taskList.join(
    ", "
  )}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.`;
  const response = await openAICall(prompt, gptVersion);
  const newTaskNames = response.split("\n");
  return newTaskNames.map((taskName) => ({ task_name: taskName }));
}

async function prioritizationAgent(thisTaskId, gpt_version = "gpt-3") {
  const taskNames = taskList.map((t) => t.task_name);
  const nextTaskId = thisTaskId + 1;
  const prompt = `You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: ${taskNames}. Consider the ultimate objective of your team: ${OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
  #. First task
  #. Second task
  Start the task list with number ${nextTaskId}.`;
  const response = await openAICall(prompt, gpt_version);
  const newTasks = response.split("\n");
  taskList.length = 0;
  newTasks.forEach((taskString) => {
    const taskParts = taskString.trim().split(".", 1);
    if (taskParts.length === 2) {
      const taskId = parseInt(taskParts[0].trim(), 10);
      const taskName = taskParts[1].trim();
      taskList.push({ task_id: taskId, task_name: taskName });
    }
  });
}

async function contextAgent(query, indexName, n) {
  const queryEmbedding = await get_ada_embedding(query);
  const results = await index.query(queryEmbedding, n, {
    includeMetadata: true,
  });
  const sortedResults = results.matches.sort((a, b) => b.score - a.score);
  return sortedResults.map((item) => item.metadata.task);
}

async function executionAgent(objective, task, gpt_version = "gpt-3") {
  const context = await contextAgent(index, objective, 5);
  const prompt = `You are an AI who performs one task based on the following objective: ${objective}.\nTake into account these previously completed tasks: ${context}\nYour task: ${task}\nResponse:`;
  return openAICall(prompt, gpt_version, 0.7, 2000);
}

addTask({ task_id: 1, task_name: YOUR_FIRST_TASK });

async function mainLoop() {
  for (let taskIdCounter = 0; ; taskIdCounter++) {
    let thisTaskId;
    let enrichedResult;
    let task;

    if (taskList.length > 0) {
      console.log("\x1b[95m\x1b[1m\n*****TASK LIST*****\n\x1b[0m\x1b[0m");
      taskList.forEach((t) => console.log(`${t.task_id}: ${t.task_name}`));

      task = taskList.shift();
      console.log("\x1b[92m\x1b[1m\n*****NEXT TASK*****\n\x1b[0m\x1b[0m");
      console.log(`${task.task_id}: ${task.task_name}`);

      const result = await executionAgent(OBJECTIVE, task.task_name);
      thisTaskId = task.task_id;
      console.log("\x1b[93m\x1b[1m\n*****TASK RESULT*****\n\x1b[0m\x1b[0m");
      console.log(result);

      // Step 2: Enrich result and store in Pinecone
      enrichedResult = { data: result }; // This is where you should enrich the result if needed
      const resultId = `result_${task.task_id}`;
      const vector = enrichedResult.data; // extract the actual result from the dictionary
      await index.upsert([
        {
          id: resultId,
          vector: await get_ada_embedding(vector),
          metadata: { task: task.task_name, result },
        },
      ]);
    }

    // Step 3: Create new tasks and reprioritize task list
    const newTasks = await taskCreationAgent(
      OBJECTIVE,
      enrichedResult,
      task.task_name,
      taskList.map((t) => t.task_name)
    );

    for (const newTask of newTasks) {
      taskIdCounter += 1;
      newTask.task_id = taskIdCounter;
      addTask(newTask);
    }

    await prioritizationAgent(thisTaskId);
  }
}

// Sleep before checking the task list again
await setTimeout(mainLoop, 1000);
