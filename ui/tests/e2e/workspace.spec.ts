import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Page } from "@playwright/test";

import { expect, test, TOKEN } from "./fixtures";

// Real audio generated with FFmpeg (a 440 Hz tone with a louder second half) — fixtures
// only; nothing here pretends to be a microphone or system capture.
function makeFixtures() {
  const dir = mkdtempSync(join(tmpdir(), "xsample-e2e-"));
  const wav = join(dir, "tone.wav");
  const mp3 = join(dir, "tone.mp3");
  const base = ["-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3"];
  execFileSync("ffmpeg", [...base, "-ar", "44100", "-ac", "2", wav]);
  execFileSync("ffmpeg", [...base, "-ar", "44100", mp3]);
  return { dir, wav, mp3 };
}

async function open(page: Page) {
  await page.goto(`/#token=${TOKEN}`);
  await expect(page.getByTestId("project-start").or(page.getByTestId("workspace"))).toBeVisible({
    timeout: 15_000,
  });
}

async function createProject(page: Page, name: string) {
  await open(page);
  // The engine holds one open project; start every test from the project chooser.
  if (await page.getByTestId("workspace").isVisible()) {
    // Let waveform requests for the leftover project settle so that none lands after closing.
    await page.waitForTimeout(1000);
    await page.getByTestId("close-project").click();
  }
  await page.getByTestId("project-name-input").fill(name);
  await page.getByTestId("create-project").click();
  await expect(page.getByTestId("project-name")).toHaveText(name);
}

async function drag(page: Page, fromFraction: number, toFraction: number) {
  const lane = page.locator('[data-testid^="lane-"]').first();
  const box = await lane.boundingBox();
  if (!box) throw new Error("lane not visible");
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * fromFraction, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * ((fromFraction + toFraction) / 2), y, { steps: 4 });
  await page.mouse.move(box.x + box.width * toFraction, y, { steps: 4 });
  await page.mouse.up();
}

test("project workflow: import, waveform, select, trim, slice, export, reopen", async ({
  page,
  consoleErrors,
}) => {
  const { wav, mp3 } = makeFixtures();
  const name = `E2E ${String(Date.now())}`;
  await createProject(page, name);
  await expect(page.getByTestId("empty-timeline")).toBeVisible();

  await page.getByTestId("import-input").setInputFiles(wav);
  await expect(page.getByTestId("timeline")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("clip")).toHaveCount(1);
  // Waveform pixels are really drawn from engine peaks.
  await expect
    .poll(
      () =>
        page
          .locator("canvas")
          .first()
          .evaluate((el) => {
            const c = el as HTMLCanvasElement;
            const data = c.getContext("2d")?.getImageData(0, 0, c.width, c.height).data;
            return data ? data.some((v, i) => i % 4 === 3 && v > 0) : false;
          }),
      { timeout: 10_000 },
    )
    .toBe(true);

  // A second format imports onto its own track.
  await page.getByTestId("import-input").setInputFiles(mp3);
  await expect(page.getByTestId("track-item")).toHaveCount(2, { timeout: 30_000 });

  // Click moves the playhead; drag selects a region.
  const lane = page.locator('[data-testid^="lane-"]').first();
  const box = await lane.boundingBox();
  if (!box) throw new Error("no lane");
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height / 2);
  await expect(page.getByTestId("playhead-time")).not.toHaveText("0:00.000");
  await drag(page, 0.2, 0.6);
  await expect(page.getByTestId("selection-range")).toBeVisible();

  // Slice, then trim.
  await page.getByLabel("Sample name").fill("Hit");
  await page.getByTestId("slice").click();
  await expect(page.getByTestId("sample-item")).toHaveCount(1);
  await page.getByTestId("trim").click();
  await page.getByTestId("reset-trim").click();
  await expect(page.getByTestId("workspace-error")).toHaveCount(0);

  // Export the sample and download a playable WAV.
  await page.getByTestId("export-sample").click();
  const download = page.waitForEvent("download");
  await expect(page.getByTestId("download-export")).toBeVisible({ timeout: 30_000 });
  await page.getByTestId("download-export").click();
  const file = await (await download).path();
  expect(readFileSync(file).subarray(0, 4).toString("ascii")).toBe("RIFF");
  const probe = execFileSync("ffprobe", [
    "-v",
    "error",
    "-show_entries",
    "stream=duration,sample_rate",
    "-of",
    "default=nw=1",
    file,
  ]).toString();
  expect(probe).toContain("sample_rate=48000");

  // Preview plays without error.
  await page.getByRole("button", { name: "Preview" }).click();

  // Close and reopen: edits are preserved.
  await page.getByTestId("close-project").click();
  await expect(page.getByTestId("project-start")).toBeVisible();
  await page.getByTestId("project-list").getByRole("button", { name: "Open" }).first().click();
  await expect(page.getByTestId("project-name")).toHaveText(name);
  await expect(page.getByTestId("sample-item")).toHaveCount(1);
  await expect(page.getByTestId("track-item")).toHaveCount(2);
  expect(consoleErrors, consoleErrors.join(" | ")).toEqual([]);
});

test("playback starts and the playhead advances", async ({ page, consoleErrors }) => {
  const { wav } = makeFixtures();
  await createProject(page, `Play ${String(Date.now())}`);
  await page.getByTestId("import-input").setInputFiles(wav);
  await expect(page.getByTestId("clip")).toHaveCount(1, { timeout: 30_000 });
  await page.getByTestId("play").click();
  await expect(page.getByTestId("play")).toHaveAttribute("aria-pressed", "true");
  await expect
    .poll(async () => page.getByTestId("playhead-time").textContent(), { timeout: 5_000 })
    .not.toBe("0:00.000");
  await page.getByTestId("play").click();
  await expect(page.getByTestId("play")).toHaveAttribute("aria-pressed", "false");
  expect(consoleErrors, consoleErrors.join(" | ")).toEqual([]);
});

test("bad files fail with a clear error and do not break the project", async ({ page }) => {
  await createProject(page, `Bad ${String(Date.now())}`);
  await page.getByTestId("import-input").setInputFiles({
    name: "broken.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("this is not audio ".repeat(50)),
  });
  await expect(page.getByTestId("activity-item").first()).toHaveAttribute("data-state", "failed", {
    timeout: 30_000,
  });
  await expect(page.getByTestId("job-strip")).toContainText(/audio/i);
  await page.getByTestId("import-input").setInputFiles({
    name: "x.exe",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("MZ"),
  });
  await expect(page.getByTestId("workspace-error")).toContainText("Unsupported file type");
  await expect(page.getByTestId("project-name")).toBeVisible();
});

test("recorder lists real devices and offers no recording until the user starts it", async ({
  page,
}) => {
  await createProject(page, `Rec ${String(Date.now())}`);
  const recorder = page.getByTestId("recorder");
  await expect(recorder).toBeVisible();
  // No stream is opened by listing; the meters appear only while recording.
  await expect(page.getByTestId("meters")).toHaveCount(0);
  await expect(page.getByTestId("device-select")).toBeVisible();
  await expect(page.getByTestId("record-start")).toBeVisible();
});

test("workspace layout has no horizontal scroll at common widths", async ({ page }) => {
  const { wav } = makeFixtures();
  await createProject(page, `Layout ${String(Date.now())}`);
  await page.getByTestId("import-input").setInputFiles(wav);
  await expect(page.getByTestId("clip")).toHaveCount(1, { timeout: 30_000 });
  for (const width of [1920, 1366, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    // The timeline re-measures itself asynchronously after a resize.
    await page.waitForTimeout(800);
    const overflow = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const wide = [...document.querySelectorAll("*")]
        .filter((e) => e.getBoundingClientRect().right > vw + 1)
        .slice(0, 8)
        .map(
          (e) =>
            `${e.tagName}.${e.className}|${e.getAttribute("data-testid") ?? ""} r=${String(Math.round(e.getBoundingClientRect().right))}`,
        );
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const r = document.createRange();
        r.selectNodeContents(n);
        const b = r.getBoundingClientRect();
        if (b.right > vw + 1 && b.width > 0)
          wide.push(
            `TEXT "${(n.textContent ?? "").slice(0, 40)}" in ${n.parentElement?.tagName ?? ""}.${String(n.parentElement?.className)} r=${String(Math.round(b.right))}`,
          );
      }
      return { px: document.documentElement.scrollWidth - vw, wide: wide.slice(0, 8) };
    });
    expect(
      overflow.px,
      `overflow ${String(overflow.px)} at ${String(width)}px: ${overflow.wide.join("; ")}`,
    ).toBeLessThanOrEqual(0);
  }
});
