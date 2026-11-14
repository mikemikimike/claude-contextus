import path from 'node:path';
import { projectHash } from '../config.js';

/** Canonical embedding context for a project-scoped write or reconcile. */
export interface ProjectEmbeddingContext {
  projectId: string;
  cwd: string;
  scope: 'project';
}

/**
 * Resolve the values that every project-scoped embedding entry point must pass
 * through to the embedding and storage layers.
 */
export function resolveProjectEmbeddingContext(cwd: string): ProjectEmbeddingContext {
  const absoluteCwd = path.resolve(cwd);
  return {
    projectId: projectHash(absoluteCwd),
    cwd: absoluteCwd,
    scope: 'project'
  };
}
