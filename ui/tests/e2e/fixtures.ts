import { test as base, expect } from "@playwright/test";

import { E2E_PORT, E2E_TOKEN } from "../../playwright.config";

export { expect };
export const TOKEN = E2E_TOKEN;
export const ORIGIN = `http://127.0.0.1:${E2E_PORT}`;

// Every page is network-isolated: only the local engine may be contacted, and any
// console error (including CSP violations) fails the test.
export const test = base.extend<{ blocked: string[]; consoleErrors: string[] }>({
  blocked: async ({ context }, use) => {
    const blocked: string[] = [];
    await context.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.origin === ORIGIN) {
        await route.continue();
      } else {
        blocked.push(url.href);
        await route.abort("blockedbyclient");
      }
    });
    await use(blocked);
    expect(blocked, "page attempted to contact non-local hosts").toEqual([]);
  },
  consoleErrors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("response", (r) => {
      if (r.status() === 409) errors.push(`409 ${r.request().method()} ${r.url()}`);
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await use(errors);
  },
  page: async ({ page, blocked: _blocked }, use) => {
    await use(page);
  },
});
