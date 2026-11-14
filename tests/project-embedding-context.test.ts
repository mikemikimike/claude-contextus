import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { projectHash } from '../lib/config.js';
import { resolveProjectEmbeddingContext } from '../lib/embedding/project-context.js';

test('resolveProjectEmbeddingContext keeps project id and cwd canonical', () => {
  const input = path.join(process.cwd(), 'test-fixtures', '..', 'project');
  const expectedCwd = path.resolve(input);
  const context = resolveProjectEmbeddingContext(input);

  assert.deepEqual(context, {
    projectId: projectHash(expectedCwd),
    cwd: expectedCwd,
    scope: 'project'
  });
});
