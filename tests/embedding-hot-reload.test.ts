import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { embedDocuments } from '../lib/embedding/embedder.js';
import { resetConfigCache } from '../lib/config.js';

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
