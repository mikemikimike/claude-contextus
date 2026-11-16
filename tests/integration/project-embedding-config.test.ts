import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';

function runNode(args: string[], cwd: string, env: NodeJS.ProcessEnv): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd, env, windowsHide: true });
    let output = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => (output += chunk));
    child.stderr.on('data', (chunk: string) => (output += chunk));
    child.once('error', reject);
    child.once('close', (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(`Child exited with ${code}:\n${output}`));
    });
  });
}

async function startEmbeddingServer(): Promise<{
  server: ReturnType<typeof createServer>;
  endpoint: string;
  requests: string[];
}> {
  const requests: string[] = [];
  const server = createServer((req, res) => {
    requests.push(req.url ?? '');
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ embedding: [1, 1] }));
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  return { server, endpoint: `http://127.0.0.1:${address.port}`, requests };
}

function writeGlobalConfig(home: string, endpoint: string): string {
  const pluginRoot = path.join(home, 'plugin');
  fs.mkdirSync(pluginRoot, { recursive: true });
  fs.writeFileSync(
    path.join(pluginRoot, 'plugin.json'),
    JSON.stringify({
      config: {
        embedding: { provider: 'ollama', ollamaModel: 'test-model', dimensions: 2 },
        providers: { ollama: { endpoint: `${endpoint}/global` } }
      }
    })
  );
  return pluginRoot;
}

function writeContextFile(file: string, body: string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(
    file,
    `---\nname: test\ndescription: embedding config test\nmetadata:\n  type: decision\n---\n${body}`
  );
}

test('reconcile and direct incremental embeds use project config while shared rows use global config', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-project-embed-home-'));
  const project = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-project-embed-cwd-'));
  const { server, endpoint, requests } = await startEmbeddingServer();
  const oldHome = process.env.HOME;
  const oldUserProfile = process.env.USERPROFILE;
  const oldPluginRoot = process.env.CLAUDE_PLUGIN_ROOT;
  const oldFakeEmbed = process.env.CONTEXTUS_TEST_FAKE_EMBED;
  const pluginRoot = writeGlobalConfig(home, endpoint);
  const contextDir = path.join(project, '.claude', 'context');
  const contextFile = path.join(contextDir, 'decision-test.md');
  fs.mkdirSync(path.join(project, '.claude'), { recursive: true });
  fs.writeFileSync(
    path.join(project, '.claude', 'contextus.json'),
    JSON.stringify({ providers: { ollama: { endpoint: `${endpoint}/project` } } })
  );
  writeContextFile(contextFile, 'initial project context');

  process.env.HOME = home;
  process.env.USERPROFILE = home;
  process.env.CLAUDE_PLUGIN_ROOT = pluginRoot;
  delete process.env.CONTEXTUS_TEST_FAKE_EMBED;
  const { projectHash, resetConfigCache } = await import('../../lib/config.js');
  const { embedSingleFile } = await import('../../lib/embedding/embed-file.js');
  const { runReconcile } = await import('../../lib/embedding/reconcile-run.js');
  const { resetConnection } = await import('../../lib/storage/lance-client.js');
  resetConfigCache();
  resetConnection();

  try {
    const projectId = projectHash(project);
    const projectName = path.basename(project);
    const reconciled = await runReconcile(
      contextDir,
      projectName,
      projectId,
      'projectContext',
      undefined,
      'context',
      project,
      'project'
    );
    assert.equal(reconciled.embedded, 1);
    assert.equal(requests.at(-1), '/project/api/embeddings');

    writeContextFile(contextFile, 'updated project context');
    const incremental = await embedSingleFile(
      contextFile,
      contextDir,
      projectName,
      'projectContext',
      projectId,
      'context',
      project,
      'project'
    );
    assert.equal(incremental.embedded, true);
    assert.equal(requests.at(-1), '/project/api/embeddings');

    const sharedDocs = path.join(home, '.contextus', 'docs');
    writeContextFile(path.join(sharedDocs, 'guide.md'), 'shared user context');
    const shared = await runReconcile(
      sharedDocs,
      projectName,
      '',
      'userDocs',
      undefined,
      'doc',
      project,
      'user'
    );
    assert.equal(shared.embedded, 1);
    assert.equal(requests.at(-1), '/global/api/embeddings');
  } finally {
    resetConnection();
    resetConfigCache();
    if (oldHome === undefined) delete process.env.HOME;
    else process.env.HOME = oldHome;
    if (oldUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = oldUserProfile;
    if (oldPluginRoot === undefined) delete process.env.CLAUDE_PLUGIN_ROOT;
    else process.env.CLAUDE_PLUGIN_ROOT = oldPluginRoot;
    if (oldFakeEmbed === undefined) delete process.env.CONTEXTUS_TEST_FAKE_EMBED;
    else process.env.CONTEXTUS_TEST_FAKE_EMBED = oldFakeEmbed;
    await new Promise<void>((resolve, reject) =>
      server.close((err) => (err ? reject(err) : resolve()))
    );
    fs.rmSync(home, { recursive: true, force: true });
    fs.rmSync(project, { recursive: true, force: true });
  }
});

test('full rebuild embeds project and shared sources with their respective config snapshots', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-rebuild-embed-home-'));
  const project = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-rebuild-embed-cwd-'));
  const { server, endpoint, requests } = await startEmbeddingServer();
  const oldHome = process.env.HOME;
  const oldUserProfile = process.env.USERPROFILE;
  const oldPluginRoot = process.env.CLAUDE_PLUGIN_ROOT;
  const pluginRoot = writeGlobalConfig(home, endpoint);
  const projectContextDir = path.join(project, '.claude', 'context');
  const sharedDocs = path.join(home, '.contextus', 'docs');
  writeContextFile(path.join(projectContextDir, 'decision-project.md'), 'project source');
  writeContextFile(path.join(sharedDocs, 'guide.md'), 'shared source');
  fs.mkdirSync(path.join(project, '.claude'), { recursive: true });
  fs.writeFileSync(
    path.join(project, '.claude', 'contextus.json'),
    JSON.stringify({
      providers: { ollama: { endpoint: `${endpoint}/project` } },
      sources: {
        projectContext: {
          enabled: true,
          scope: 'project',
          type: 'context',
          path: '.claude/context'
        },
        projectDocs: { enabled: false },
        userContext: { enabled: false },
        userDocs: { enabled: true, scope: 'user', type: 'doc', path: sharedDocs }
      }
    })
  );

  try {
    const script = path.join(process.cwd(), 'scripts', 'embed-contexts.ts');
    const output = await runNode(['--import', import.meta.resolve('tsx'), script], project, {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
      CLAUDE_PLUGIN_ROOT: pluginRoot,
      CONTEXTUS_TEST_FAKE_EMBED: '0'
    });
    assert.match(output, /Embedded/);
    assert.ok(requests.includes('/project/api/embeddings'));
    assert.ok(requests.includes('/global/api/embeddings'));
  } finally {
    if (oldHome === undefined) delete process.env.HOME;
    else process.env.HOME = oldHome;
    if (oldUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = oldUserProfile;
    if (oldPluginRoot === undefined) delete process.env.CLAUDE_PLUGIN_ROOT;
    else process.env.CLAUDE_PLUGIN_ROOT = oldPluginRoot;
    await new Promise<void>((resolve, reject) =>
      server.close((err) => (err ? reject(err) : resolve()))
    );
    fs.rmSync(home, { recursive: true, force: true });
    fs.rmSync(project, { recursive: true, force: true });
  }
});
