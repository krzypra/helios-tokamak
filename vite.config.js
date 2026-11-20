import { defineConfig } from 'vite';

export default defineConfig({
  // Use relative asset paths so the built dist/ folder works at root domains and subpaths.
  base: './',
});
