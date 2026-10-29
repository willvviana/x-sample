import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Page } from "@playwright/test";

import { expect, test, TOKEN } from "./fixtures";

// Real audio rendered by FFmpeg: a decaying 60 Hz kick every 0.5 s (120 BPM) over a sustained
// C major triad. Tempo and key are therefore known exactly.
function makeGroove(): string {
  const dir = mkdtempSync(join(tmpdir(), "xsample-e2e-analysis-"));
  const wav = join(dir, "groove 120 C.wav");
  const kick = "0.8*exp(-30*mod(t,0.5))*sin(2*PI*60*t)";
  const triad = "0.12*(sin(2*PI*261.63*t)+sin(2*PI*329.63*t)+sin(2*PI*392*t))";
  execFileSync("ffmpeg", [
    "-y", "-loglevel", "error", "-f", "lavfi",
    "-i", `aevalsrc='${kick}+${triad}':s=44100:d=16`,
    "-ac", "2", wav,
  ]); // prettier-ignore
  return wav;
}

async function createProject(page: Page, name: string) {
  await page.goto(`/#token=${TOKEN}`);
  await expect(page.getByTestId("project-start").or(page.getByTestId("workspace"))).toBeVisible({
    timeout: 15_000,
  });
  if (await page.getByTestId("workspace").isVisible()) {
    await page.waitForTimeout(1000);
    await page.getByTestId("close-project").click();
  }
  await page.getByTestId("project-name-input").fill(name);
  await page.getByTestId("create-project").click();
  await expect(page.getByTestId("project-name")).toHaveText(name);
}

function bpmOf(text: string | null): number {
  return Number((text ?? "").replace(/[^\d.]/g, ""));
}

test("analysis: run, beat grid on the waveform, corrections, tags, reset", async ({
  page,
  consoleErrors,
}) => {
  await createProject(page, `Analysis ${String(Date.now())}`);
  await page.getByTestId("import-input").setInputFiles(makeGroove());
  await expect(page.getByTestId("clip")).toHaveCount(1, { timeout: 30_000 });

  const panel = page.getByTestId("analysis-panel");
  await expect(panel.getByTestId("analysis-status")).toHaveText("Not analysed yet.");
  await panel.getByTestId("analysis-run").click();
  await expect(panel.getByTestId("analysis-summary")).toBeVisible({ timeout: 60_000 });

  // Measured values, not decoration: the fixture is 120 BPM in C major.
  expect(Math.abs(bpmOf(await panel.getByTestId("analysis-bpm").textContent()) - 120)).toBeLessThan(
    2,
  );
  await expect(panel.getByTestId("analysis-key")).toHaveText("C major");
  await expect(panel.getByTestId("analysis-backend")).toHaveText("CPU");
  await expect(panel.getByText("Genre recognition:")).toBeVisible();
  await expect(page.getByTestId("beat-grid").locator(".beat-line").first()).toBeAttached();
  await expect(panel.getByTestId("analysis-canvas")).toBeVisible();

  // Half/double-time correction changes the effective tempo, not the detected one.
  await panel.getByRole("button", { name: "Double time" }).click();
  await expect
    .poll(async () => bpmOf(await panel.getByTestId("analysis-bpm").textContent()))
    .toBeGreaterThan(230);
  await expect(panel.getByTestId("analysis-corrected")).toContainText("beat_grid");

  // Key correction and manual tags.
  await panel.getByTestId("key-tonic").selectOption("9");
  await panel.getByTestId("key-mode").selectOption("minor");
  await expect(panel.getByTestId("analysis-key")).toHaveText("A minor");
  await panel.getByTestId("tag-input").fill("R&B");
  await panel.getByRole("button", { name: "Add tag" }).click();
  await expect(panel.getByTestId("tag-list")).toContainText("R&B");

  // Chord correction through the keyboard-accessible chord list.
  await panel.getByText(/Chord list/).click();
  await panel.getByTestId("chord-list").getByRole("button").first().click();
  await panel.getByTestId("chord-root").selectOption("5");
  await panel.getByTestId("chord-quality").selectOption("maj7");
  await panel.getByTestId("chord-editor").getByRole("button", { name: "Apply" }).click();
  await expect(panel.getByTestId("chord-list")).toContainText("Fmaj7");

  // Corrections survive a re-run (they live in the project, not in the analysis cache).
  await panel.getByTestId("analysis-run").click();
  await expect(panel.getByTestId("analysis-run")).toBeEnabled({ timeout: 60_000 });
  await expect(panel.getByTestId("analysis-key")).toHaveText("A minor");

  // The filled panel must not widen the page (the lanes scroll inside their own box).
  await panel.getByText(/Chord list/).click();
  for (const width of [1366, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForTimeout(600);
    const overflow = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const scrollers = [...document.querySelectorAll(".analysis-scroll")];
      const wide = [...document.querySelectorAll("*")]
        .filter((e) => !scrollers.some((s) => s !== e && s.contains(e)))
        .filter((e) => e.getBoundingClientRect().right > vw + 1)
        .map((e) => `${e.tagName}.${e.className}`);
      return { px: document.documentElement.scrollWidth - vw, wide: wide.slice(0, 6) };
    });
    expect(overflow.px, `${String(width)}px: ${overflow.wide.join("; ")}`).toBeLessThanOrEqual(0);
  }
  await page.setViewportSize({ width: 1366, height: 900 });

  await panel.getByTestId("analysis-reset").click();
  await expect(panel.getByTestId("analysis-key")).toHaveText("C major");
  await expect(panel.getByTestId("tag-list")).not.toContainText("R&B");
  expect(Math.abs(bpmOf(await panel.getByTestId("analysis-bpm").textContent()) - 120)).toBeLessThan(
    2,
  );
  expect(consoleErrors).toEqual([]);
});
