import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { embedDocuments, snapshotEmbeddingContext } from '../lib/embedding/embedder.js';
import { loadConfig, loadProjectConfig, projectHash, resetConfigCache } from '../lib/config.js';

test('a batch keeps its embedding model when global config reloads', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-reload-'));
  const oldRoot = process.env.CLAUDE_PLUGIN_ROOT;
  const oldFetch = globalThis.fetch;
  const models: string[] = [];
  const writeConfig = (model: string) =>
    fs.writeFileSync(
      path.join(root, 'plugin.json'),
      JSON.stringify({
        config: { embedding: { provider: 'ollama', ollamaModel: model, dimensions: 2 } }
      })
    );
  process.env.CLAUDE_PLUGIN_ROOT = root;
  writeConfig('first');
  resetConfigCache();
  globalThis.fetch = async (_input, init) => {
    models.push(JSON.parse(String(init?.body)).model);
    if (models.length === 1) {
      writeConfig('second');
      resetConfigCache();
    }
    return new Response(JSON.stringify({ embedding: [1, 1] }), { status: 200 });
  };
  try {
    await embedDocuments(['one', 'two']);
    assert.deepEqual(models, ['first', 'first']);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldRoot === undefined) delete process.env.CLAUDE_PLUGIN_ROOT;
    else process.env.CLAUDE_PLUGIN_ROOT = oldRoot;
    resetConfigCache();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('project and shared embedding batches keep one global source snapshot across reloads', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-project-snapshot-'));
  const pluginRoot = path.join(home, 'plugin');
  const cwd = path.join(home, 'project');
  const oldHome = process.env.HOME;
  const oldProfile = process.env.USERPROFILE;
  const oldRoot = process.env.CLAUDE_PLUGIN_ROOT;
  const oldFetch = globalThis.fetch;
  const endpoints: string[] = [];
  fs.mkdirSync(pluginRoot, { recursive: true });
  fs.mkdirSync(path.join(cwd, '.claude'), { recursive: true });
  fs.writeFileSync(
    path.join(pluginRoot, 'plugin.json'),
    JSON.stringify({
      config: {
        embedding: {
          provider: 'ollama',
          ollamaModel: 'shared-model',
          dimensions: 2,
          transformers: {
            localModelPath: './global-models',
            cacheDir: './global-cache',
            remoteHost: 'http://global-mirror.test',
            allowRemoteModels: true
          }
        },
        providers: { ollama: { endpoint: 'http://global.test' } }
      }
    })
  );
  fs.writeFileSync(
    path.join(cwd, '.claude', 'contextus.json'),
    JSON.stringify({
      providers: { ollama: { endpoint: 'http://project.test' } },
      retrieval: { topK: 6 }
    })
  );
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  process.env.CLAUDE_PLUGIN_ROOT = pluginRoot;
  resetConfigCache();
  globalThis.fetch = async (input) => {
    endpoints.push(String(input));
    if (endpoints.length === 1) {
      fs.writeFileSync(
        path.join(pluginRoot, 'plugin.json'),
        JSON.stringify({
          config: {
            embedding: { provider: 'ollama', ollamaModel: 'shared-model', dimensions: 2 },
            providers: { ollama: { endpoint: 'http://reloaded.test' } }
          }
        })
      );
      resetConfigCache();
    }
    return new Response(JSON.stringify({ embedding: [1, 1] }), { status: 200 });
  };
  try {
    const projectConfig = loadProjectConfig(projectHash(cwd), cwd);
    projectConfig.embedding.transformers = {
      localModelPath: './project-models',
      cacheDir: './project-cache',
      remoteHost: 'http://project-mirror.test',
      allowRemoteModels: false
    };
    const projectContext = snapshotEmbeddingContext(projectConfig);
    const sharedContext = snapshotEmbeddingContext(loadConfig());
    assert.equal(projectContext.providers.ollama.endpoint, 'http://global.test');
    assert.equal(projectContext.embedding.provider, 'ollama');
    assert.equal(projectContext.embedding.ollamaModel, 'shared-model');
    assert.equal(projectContext.embedding.dimensions, 2);
    assert.deepEqual(projectContext.embedding.transformers, sharedContext.embedding.transformers);
    assert.equal(projectContext.embedding.transformers?.localModelPath, './global-models');
    assert.equal(projectContext.embedding.transformers?.cacheDir, './global-cache');
    assert.equal(projectContext.embedding.transformers?.remoteHost, 'http://global-mirror.test');
    assert.equal(projectContext.embedding.transformers?.allowRemoteModels, true);
    assert.equal(projectConfig.retrieval.topK, 6);
    assert.equal(sharedContext.providers.ollama.endpoint, 'http://global.test');
    assert.equal(sharedContext.embedding.transformers?.remoteHost, 'http://global-mirror.test');

    await embedDocuments(['project row', 'another project row'], undefined, projectContext);
    await embedDocuments(['shared user row'], undefined, sharedContext);

    assert.deepEqual(endpoints, Array(3).fill('http://global.test/api/embeddings'));
  } finally {
    globalThis.fetch = oldFetch;
    if (oldHome === undefined) delete process.env.HOME;
    else process.env.HOME = oldHome;
    if (oldProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = oldProfile;
    if (oldRoot === undefined) delete process.env.CLAUDE_PLUGIN_ROOT;
    else process.env.CLAUDE_PLUGIN_ROOT = oldRoot;
    resetConfigCache();
    fs.rmSync(home, { recursive: true, force: true });
  }
});
