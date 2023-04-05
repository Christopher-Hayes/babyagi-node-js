#!/usr/bin/env node
/* eslint-disable no-console */
/* eslint-disable no-await-in-loop */
import {
  Configuration,
  CreateChatCompletionRequest,
  CreateCompletionRequest,
  CreateEmbeddingResponseDataInner,
  OpenAIApi,
} from "openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { CreateIndexRequest } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";
import {
  VectorOperationsApi,
} from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";
import { v4 as uuidv4 } from "uuid";

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || "us-east1-gcp";
const YOUR_TABLE_NAME = process.env.TABLE_NAME;
const OBJECTIVE = process.argv[2] || process.env.OBJECTIVE;
const YOUR_FIRST_TASK = process.env.FIRST_TASK;

const DEBUG = false;

// Set to "true" to use gpt-4. Be aware that gpt-4 is between 15x and 30x more expensive than gpt-3.5-turbo
const GPT_VERSION: string = "gpt-3.5-turbo";

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

const configuration = new Configuration({
  // organization: "",
  apiKey: OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: PINECONE_API_KEY,
  environment: PINECONE_ENVIRONMENT,
});

const tableName = YOUR_TABLE_NAME;
const dimension = 1536;
const metric = "cosine";
const podType = "p1";

const indexList = await pinecone.listIndexes();
if (!indexList.includes(tableName)) {
  const createIndexOptions: CreateIndexRequest = {
    createRequest: {
      name: tableName,
      dimension,
      metric,
      podType,
    },
  };
  if (DEBUG) console.log("Creating index with options: ", createIndexOptions);

  await pinecone.createIndex(createIndexOptions);
}

let index: VectorOperationsApi = pinecone.Index(tableName);

interface Task {
  id: string;
  name: string;
  priority: number; // 1 is highest priority
}

let taskList: Task[] = [];
let embeddingList = new Map<string, number[]>();

async function getADAEmbedding(text: string): Promise<number[]> {
  if (DEBUG) console.log("\nGetting ADA embedding for: ", text);

  if (embeddingList.has(text)) {
    if (DEBUG) console.log("Embedding already exists for: ", text);
    return embeddingList.get(text);
  }

  const embedding = (
    await openai.createEmbedding({
      input: [text],
      model: "text-embedding-ada-002",
    })
  ).data?.data[0].embedding;

  embeddingList.set(text, embedding);

  return embedding;
}

async function openAICall(
  prompt: string,
  gptVersion: string,
  temperature = 0.5,
  max_tokens = 100
) {
  if (
    gptVersion === "gpt-3.5-turbo" ||
    gptVersion === "gpt-4" ||
    gptVersion === "gpt-4-32k"
  ) {
    // Chat completion
    const options: CreateChatCompletionRequest = {
      model: gptVersion,
      messages: [{ role: "user", content: prompt }],
      temperature,
      max_tokens,
      n: 1,
    };
    const data = (await openai.createChatCompletion(options)).data;

    return data?.choices[0]?.message?.content.trim() ?? "";
  } else {
    // Prompt completion
    const options: CreateCompletionRequest = {
      model: gptVersion,
      prompt,
      temperature,
      max_tokens,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
    };
    const data = (await openai.createCompletion(options)).data;

    return data?.choices[0]?.text?.trim() ?? "";
  }
}

async function taskCreationAgent(
  objective: string,
  result: string,
  taskDescription: string,
  gptVersion = "gpt-3.5-turbo"
): Promise<Task[]> {
  const prompt = `You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: ${objective}, The last completed task has the result: ${result}. This result was based on this task description: ${taskDescription}. These are incomplete tasks: ${taskList.join(
    ", "
  )}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.`;
  const response = await openAICall(prompt, gptVersion);
  const newTaskNames = response.split("\n");

  return newTaskNames.map((name) => ({
    id: uuidv4(),
    name,
    priority: taskList.length + 1,
  }));
}

async function prioritizationAgent(
  taskPriority: number,
  gptVersion = "gpt-3.5-turbo"
): Promise<Task[]> {
  const taskNames = taskList.map((t) => t.name);
  const startPriority = taskPriority + 1;

  const prompt = `You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: ${taskNames}. Consider the ultimate objective of your team: ${OBJECTIVE}. Do not remove any tasks. Return the result as a list, like:
#. First task
#. Second task
Start the task list with number ${startPriority}.`;
  const response = await openAICall(prompt, gptVersion);
  const newTasks = response.split("\n");

  // Parse and add new tasks
  return (
    newTasks
      .map((taskString) => {
        const taskParts = taskString.trim().split(".", 2);

        if (taskParts.length === 2) {
          const id = uuidv4();
          const name = taskParts[1].trim();
          const priority = parseInt(taskParts[0]);
          return {
            id,
            name,
            priority,
          };
        }
      })
      // Remove lines that don't have a task
      .filter((t) => t !== undefined)
      // Sort by priority
      .sort((a, b) => a.priority - b.priority)
  );
}

async function contextAgent(task: Task, objective: string, topK: number) {
  index = pinecone.Index(YOUR_TABLE_NAME);
  // const queryEmbedding = await getADAEmbedding(task.name)
  const queryEmbedding = await getADAEmbedding(objective);

  const results = await index.query({
    queryRequest: {
      vector: queryEmbedding,
      includeMetadata: true,
      topK,
    },
  });
  const sortedResults =
    results.matches?.sort((a, b) => (b?.score ?? 0) - (a?.score ?? 0)) ?? [];

  return sortedResults.map((item) => (item.metadata as any)?.task ?? "");
}

async function executionAgent(
  objective: string,
  task: Task,
  gptVersion = "gpt-3.5-turbo"
) {
  const context = await contextAgent(task, objective, 5);
  const prompt = `You are an AI who performs one task based on the following objective: ${objective}.\nTake into account these previously completed tasks: ${context}\nYour task: ${task.name}\nResponse:`;

  if (DEBUG) console.log("\nexecution prompt: ", prompt, "\n");

  return openAICall(prompt, gptVersion, 0.7, 2000);
}

async function mainLoop() {
  // Risk mitigation - limit the number of task runs
  const RUN_LIMIT = 10;
  for (let run = 0; run < RUN_LIMIT; run++) {
    let enrichedResult: { data: any };
    let task: Task | undefined;

    if (taskList.length > 0) {
      task = taskList.shift();

      if (!task) {
        console.log("No tasks left to complete. Exiting.");
        break;
      }

      console.log(`\x1b[95m\x1b[1m\n*****TASK LIST*****\n\x1b[0m\x1b[0m
${taskList.map((t) => ` ${t?.priority}. ${t?.name}`).join("\n")}
\x1b[92m\x1b[1m\n*****NEXT TASK*****\n\x1b[0m\x1b[0m
 ${task.name}`);

      const result = await executionAgent(OBJECTIVE, task);
      console.log("\x1b[93m\x1b[1m\n*****TASK RESULT*****\n\x1b[0m\x1b[0m");
      console.log(result);

      // Enrich result and store in Pinecone
      enrichedResult = { data: result }; // This is where you should enrich the result if needed
      const vector = enrichedResult.data; // extract the actual result from the dictionary
      const embeddingResult = await getADAEmbedding(vector);
      await index.upsert({
        upsertRequest: {
          vectors: [
            {
              id: task.id,
              values: embeddingResult,
              metadata: { task: task.name, result },
            },
          ],
        },
      });
    }

    // Create new tasks
    if (enrichedResult) {
      const newTasks = await taskCreationAgent(
        OBJECTIVE,
        enrichedResult.data,
        task.name
      );
      if (DEBUG) console.log("newTasks", newTasks);
      taskList = [...taskList, ...newTasks];

      // Reprioritize tasks (using ai)
      taskList = await prioritizationAgent(task.priority, GPT_VERSION);
    } else {
      break;
    }

    if (DEBUG)
      console.log(
        `Reprioritized task list: ${taskList
          .map((t) => `[${t?.priority}] ${t?.id}: ${t?.name}`)
          .join(", ")}`
      );
  }
}

async function run() {
  taskList = [
    {
      id: uuidv4(),
      name: YOUR_FIRST_TASK,
      priority: 1,
    },
  ];

  await mainLoop();
}

await run();
