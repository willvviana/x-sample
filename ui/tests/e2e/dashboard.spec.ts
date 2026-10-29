import type { Page } from "@playwright/test";

import { expect, ORIGIN, test, TOKEN } from "./fixtures";

async function openDashboard(page: Page) {
  await page.goto(`/#token=${TOKEN}`);
  await page.getByTestId("tab-system").click();
  await expect(page.getByTestId("connection")).toContainText("Engine", { timeout: 15_000 });
  await expect(page.getByTestId("detection-state")).toContainText("ready", { timeout: 30_000 });
}

test("token is required and removed from the address bar", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("connect-screen")).toBeVisible();

  await page.goto(`/#token=${TOKEN}`);
  await page.getByTestId("tab-system").click();
  await expect(page.getByTestId("system-panel")).toBeVisible();
  expect(page.url()).toBe(`${ORIGIN}/`);
  // The token survives a reload through sessionStorage, not the URL.
  await page.reload();
  await page.getByTestId("tab-system").click();
  await expect(page.getByTestId("system-panel")).toBeVisible();
});

test("wrong token returns to the connect screen", async ({ page }) => {
  await page.goto("/#token=wrong-token-0123456789");
  await expect(page.getByTestId("connect-screen")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("alert")).toContainText("rejected the token");
});

test("API rejects requests without a token or from a foreign origin", async ({ request }) => {
  expect((await request.get("/api/system")).status()).toBe(401);
  const foreign = await request.get("/api/system", {
    headers: { Authorization: `Bearer ${TOKEN}`, Origin: "https://evil.example" },
  });
  expect(foreign.status()).toBe(403);
  const ok = await request.get("/api/system", { headers: { Authorization: `Bearer ${TOKEN}` } });
  expect(ok.status()).toBe(200);
});

test("UI is served with a strict content security policy", async ({ request }) => {
  const response = await request.get("/");
  const csp = response.headers()["content-security-policy"] ?? "";
  expect(csp).toContain("script-src 'self'");
  expect(csp).toContain("frame-ancestors 'none'");
  expect(response.headers()["x-frame-options"]).toBe("DENY");
});

test("dashboard shows system and backend status without CSP errors", async ({
  page,
  consoleErrors,
}) => {
  await openDashboard(page);
  await expect(page.getByTestId("gpu").first()).toBeVisible();
  await expect(page.getByTestId("backend-cpu-state")).toHaveText("available");
  for (const id of ["directml", "vulkan"]) {
    await expect(page.getByTestId(`backend-${id}-state`)).toHaveText(/available|unavailable/);
  }
  await expect(page.getByTestId("backend-vulkan-support")).toContainText("not claimed");
  expect(consoleErrors).toEqual([]);
});

test("probes verify CPU and DirectML, and report Vulkan as unsupported", async ({ page }) => {
  await openDashboard(page);

  const cpu = page.getByTestId("backend-cpu");
  await cpu.getByRole("button", { name: /probe/i }).click();
  await expect(page.getByTestId("probe-cpu-status")).toHaveText("pass", { timeout: 30_000 });
  await expect(page.getByTestId("backend-cpu-support")).toContainText("verified by probe");

  const vulkanAvailable =
    (await page.getByTestId("backend-vulkan-state").textContent()) === "available";
  if (vulkanAvailable) {
    await page.getByTestId("backend-vulkan").getByRole("button", { name: /probe/i }).click();
    await expect(page.getByTestId("probe-vulkan-status")).toHaveText("unsupported", {
      timeout: 30_000,
    });
    await expect(page.getByTestId("backend-vulkan-support")).toContainText("not claimed");
  }

  const directmlAvailable =
    (await page.getByTestId("backend-directml-state").textContent()) === "available";
  test.skip(!directmlAvailable, "DirectML not available on this machine");
  await page.getByTestId("backend-directml").getByRole("button", { name: /probe/i }).click();
  await expect(page.getByTestId("probe-directml-status")).toHaveText("pass", { timeout: 60_000 });
  await expect(page.getByTestId("probe-directml")).toContainText("fallback: none");
});

test("smoke inference job runs to completion with metrics", async ({ page }) => {
  await openDashboard(page);
  const directmlAvailable =
    (await page.getByTestId("backend-directml-state").textContent()) === "available";
  await page.getByTestId("smoke-backend").selectOption(directmlAvailable ? "directml" : "cpu");
  await page.getByTestId("smoke-seconds").fill("30");
  await page.getByTestId("run-smoke").click();

  const job = page.locator('[data-job-kind="smoke_inference"]').first();
  await expect(job.getByTestId("job-state")).toHaveText("succeeded", { timeout: 60_000 });
  await expect(job.getByTestId("smoke-metrics")).toContainText("verified ✓");
  await expect(job.getByTestId("smoke-metrics")).toContainText("realtime");
});

test("a running job shows progress and can be cancelled", async ({ page }) => {
  await openDashboard(page);
  await page.getByTestId("run-test-job").click();

  const job = page.locator('[data-job-kind="test.steps"]').first();
  await expect(job.getByTestId("job-state")).toHaveText("running", { timeout: 15_000 });
  await expect(job.getByRole("progressbar")).toBeVisible();
  await job.getByTestId("cancel-job").click();
  await expect(job.getByTestId("job-state")).toHaveText("cancelled", { timeout: 15_000 });
});

test("layout fits common desktop widths without horizontal scrolling", async ({ page }) => {
  for (const width of [1024, 1366, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `horizontal overflow at ${width}px`).toBeLessThanOrEqual(0);
  }
});
