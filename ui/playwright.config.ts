import { defineConfig, devices } from "@playwright/test";

export const E2E_PORT = 8766;
export const E2E_TOKEN = "e2e-token-not-secret-0123456789abcdef";

// The engine serves the production build (same origin, strict CSP), with test-only job
// kinds enabled and a throwaway data directory.
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${E2E_PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // Fresh data dir each run so probes are executed, not served from a previous cache.
    command: [
      `node -e "require('fs').rmSync('.e2e-data',{recursive:true,force:true})"`,
      "npm run build",
      `uv run --project ../engine xsample serve --port ${E2E_PORT} --ui-dist dist --data-dir .e2e-data --projects-dir .e2e-data/projects --enable-test-tasks --log-level warning`,
    ].join(" && "),
    url: `http://127.0.0.1:${E2E_PORT}/api/health`,
    env: { XSAMPLE_TOKEN: E2E_TOKEN },
    reuseExistingServer: false,
    timeout: 180_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
