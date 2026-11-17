import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DAEMON_TS = path.resolve(__dirname, '..', '..', 'daemon', 'server.ts');

async function listen(server: ReturnType<typeof createServer>): Promise<number> {
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  return address.port;
}

async function close(server: ReturnType<typeof createServer>): Promise<void> {
  await new Promise<void>((resolve, reject) =>
    server.close((err) => (err ? reject(err) : resolve()))
  );
}

async function waitForReady(daemonUrl: string, daemon: ChildProcess): Promise<void> {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (daemon.exitCode !== null) throw new Error(`daemon exited with code ${daemon.exitCode}`);
    try {
      const response = await fetch(`${daemonUrl}/health`, { signal: AbortSignal.timeout(1000) });
      const health = (await response.json()) as { modelLoaded?: boolean };
      if (response.ok && health.modelLoaded) return;
    } catch {
      // The daemon may not have bound its port yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('daemon did not finish its embedding warmup');
}

test('a project Ollama endpoint override cannot fail embeddings or clear global readiness', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-query-readiness-home-'));
  const project = fs.mkdtempSync(path.join(os.tmpdir(), 'contextus-query-readiness-project-'));
  const pluginRoot = path.join(home, 'plugin');
  fs.mkdirSync(pluginRoot, { recursive: true });

  let projectRequests = 0;
  let globalEmbeddingRequests = 0;
  const embeddingServer = createServer((req, res) => {
    if (req.url?.startsWith('/project/')) {
      projectRequests++;
      res.writeHead(503);
      res.end('project endpoint unavailable');
      return;
    }
    if (req.url?.endsWith('/api/embeddings')) {
      globalEmbeddingRequests++;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ embedding: [1, 0] }));
      return;
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{}');
  });

  let daemon: ChildProcess | undefined;
  let daemonOutput = '';
  let resetConnection: (() => void) | undefined;
  let embeddingServerListening = false;
  const oldHome = process.env.HOME;
  const oldUserProfile = process.env.USERPROFILE;
  const oldPluginRoot = process.env.CLAUDE_PLUGIN_ROOT;
  try {
    const embeddingPort = await listen(embeddingServer);
    embeddingServerListening = true;
    const portProbe = createServer();
    const daemonPort = await listen(portProbe);
    await close(portProbe);
    const embeddingUrl = `http://127.0.0.1:${embeddingPort}`;
    const daemonUrl = `http://127.0.0.1:${daemonPort}`;
    fs.writeFileSync(
      path.join(pluginRoot, 'plugin.json'),
      JSON.stringify({
        config: {
          embedding: { provider: 'ollama', ollamaModel: 'test-model', dimensions: 2 },
          providers: { ollama: { endpoint: `${embeddingUrl}/global` } },
          expansion: { mode: 'off' }
        }
      })
    );
    fs.mkdirSync(path.join(project, '.claude'), { recursive: true });
    fs.writeFileSync(
      path.join(project, '.claude', 'contextus.json'),
      JSON.stringify({ providers: { ollama: { endpoint: `${embeddingUrl}/project` } } })
    );

    process.env.HOME = home;
    process.env.USERPROFILE = home;
    process.env.CLAUDE_PLUGIN_ROOT = pluginRoot;
    const { projectHash } = await import('../../lib/config.js');
    const lanceClient = await import('../../lib/storage/lance-client.js');
    resetConnection = lanceClient.resetConnection;
    const projectId = projectHash(project);
    await lanceClient.createTable(
      [
        {
          id: `${projectId}:decision-test.md:0`,
          project: projectId,
          vector: [1, 0],
          document: 'shared and project embedding source test',
          filename: 'decision-test.md',
          chunk_index: 0,
          heading: '',
          type: 'decision',
          source: 'projectContext',
          tags: '',
          hit_count: 0,
          last_hit: '',
          updated_at: new Date().toISOString(),
          content_hash: 'readiness-test'
        }
      ],
      { providerInfo: { provider: 'ollama', model: 'test-model', ollamaModel: 'test-model' } }
    );

    const childEnv = {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
      CLAUDE_PLUGIN_ROOT: pluginRoot,
      CONTEXTUS_DAEMON_PORT: String(daemonPort)
    };
    delete childEnv.CONTEXTUS_TEST_FAKE_EMBED;
    const daemonUrlForImport = pathToFileURL(DAEMON_TS).href;
    const daemonEntryPath = daemonUrlForImport.slice('file://'.length);
    const bootScript =
      `process.argv[1] = ${JSON.stringify(daemonEntryPath)}; ` +
      `await import(${JSON.stringify(daemonUrlForImport)});`;
    daemon = spawn(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', bootScript], {
      env: childEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    });
    daemon.stdout?.on('data', (chunk: Buffer) => (daemonOutput += chunk.toString()));
    daemon.stderr?.on('data', (chunk: Buffer) => (daemonOutput += chunk.toString()));
    try {
      await waitForReady(daemonUrl, daemon);
    } catch (err) {
      throw new Error(
        `${err instanceof Error ? err.message : String(err)}; ` +
          `global embedding requests: ${globalEmbeddingRequests}; daemon output: ${daemonOutput}`
      );
    }

    const response = await fetch(`${daemonUrl}/query`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'find project context', project: projectId, cwd: project })
    });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).continue, true);

    const healthResponse = await fetch(`${daemonUrl}/health`);
    const health = (await healthResponse.json()) as {
      modelLoaded: boolean;
      warmupError: string | null;
    };
    assert.equal(health.modelLoaded, true);
    assert.equal(health.warmupError, null);
    assert.equal(projectRequests, 0, 'embedding requests must use the global endpoint');

    const connection = await lanceClient.getConnection();
    await connection.dropTable('contextus');
    await connection.createTable('contextus', [
      {
        id: `${projectId}:decision-test.md:0`,
        project: projectId,
        vector: [1, 0, 0],
        document: 'find project context after vector retrieval fails',
        filename: 'decision-test.md',
        chunk_index: 0,
        heading: '',
        type: 'decision',
        source: 'projectContext',
        tags: '',
        hit_count: 0,
        last_hit: '',
        updated_at: new Date().toISOString(),
        content_hash: 'readiness-search-failure-test'
      }
    ]);

    const searchFailureResponse = await fetch(`${daemonUrl}/query`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'find project context', project: projectId, cwd: project })
    });
    assert.equal(searchFailureResponse.status, 200);
    const errorLog = fs.readFileSync(path.join(home, '.contextus', 'logs', 'errors.jsonl'), 'utf8');
    assert.match(errorLog, /Vector retrieval failed, attempting keyword fallback/);

    const recoveredHealthResponse = await fetch(`${daemonUrl}/health`);
    const recoveredHealth = (await recoveredHealthResponse.json()) as {
      modelLoaded: boolean;
      warmupError: string | null;
    };
    assert.equal(recoveredHealth.modelLoaded, true);
    assert.equal(recoveredHealth.warmupError, null);
    assert.ok(
      globalEmbeddingRequests >= 3,
      'the global endpoint must handle warmup and both queries'
    );
  } finally {
    if (daemon && daemon.exitCode === null) {
      const exited = new Promise<void>((resolve) => daemon!.once('exit', () => resolve()));
      daemon.kill();
      await exited;
    }
    resetConnection?.();
    if (embeddingServerListening) await close(embeddingServer);
    if (oldHome === undefined) delete process.env.HOME;
    else process.env.HOME = oldHome;
    if (oldUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = oldUserProfile;
    if (oldPluginRoot === undefined) delete process.env.CLAUDE_PLUGIN_ROOT;
    else process.env.CLAUDE_PLUGIN_ROOT = oldPluginRoot;
    fs.rmSync(home, { recursive: true, force: true });
    fs.rmSync(project, { recursive: true, force: true });
  }
});
