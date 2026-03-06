#!/usr/bin/env node
/**
 * Generate PNG from the Harness Configuration Flowchart HTML file
 * Saves Harness_Configuration_Flowchart.png in the project root
 */

const { chromium } = require("playwright");
const path = require("path");

async function main() {
  const projectRoot = path.resolve(__dirname, "..");
  const htmlFile = path.join(projectRoot, "Harness_Configuration_Flowchart.html");
  const outputFile = path.join(projectRoot, "Harness_Configuration_Flowchart.png");

  console.log("📷 Generating harness flowchart PNG from:", htmlFile);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    // Set viewport to ensure the diagram fits (adjust width/height as needed)
    await page.setViewportSize({ width: 1600, height: 2400 });

    // Load the HTML file
    await page.goto(`file://${htmlFile}`, {
      waitUntil: "networkidle",
      timeout: 60000,
    });

    // Wait for mermaid to render
    await page.waitForSelector(".mermaid", { timeout: 30000 });

    // Give extra time for the diagram to fully render
    await page.waitForTimeout(5000);

    // Take screenshot of the mermaid container
    const mermaidElement = await page.$(".mermaid");
    if (mermaidElement) {
      await mermaidElement.screenshot({ path: outputFile, type: "png" });
      console.log(`✅ Saved ${outputFile}`);
    } else {
      // Fallback: screenshot the whole page
      await page.screenshot({ path: outputFile, fullPage: true });
      console.log(
        `⚠️ Saved full page screenshot to ${outputFile} (mermaid container not found)`
      );
    }
  } catch (err) {
    console.error("❌ Failed to generate harness flowchart PNG:", err.message);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  main();
}