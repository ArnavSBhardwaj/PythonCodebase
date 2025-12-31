import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const KB_DIR = path.join(process.cwd(), "kb");
const OUT = path.join(process.cwd(), "kb_index.json");

function chunkText(text, chunkSize = 900, overlap = 150) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    chunks.push(text.slice(i, i + chunkSize));
    i += chunkSize - overlap;
  }
  return chunks;
}

async function embed(texts) {
  const res = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
  });
  return res.data.map(d => d.embedding);
}

async function main() {
  const files = fs.readdirSync(KB_DIR).filter(f => f.endsWith(".txt"));
  if (!files.length) {
    console.error("❌ No .txt files found in /kb");
    process.exit(1);
  }

  const items = [];

  for (const file of files) {
    const full = fs.readFileSync(path.join(KB_DIR, file), "utf8");
    const chunks = chunkText(full);

    const vectors = await embed(chunks);

    chunks.forEach((chunk, idx) => {
      items.push({
        id: `${file}::${idx}`,
        source: file,
        text: chunk,
        embedding: vectors[idx],
      });
    });
  }

  fs.writeFileSync(OUT, JSON.stringify({ items }, null, 2));
  console.log(`✅ Wrote ${items.length} chunks -> ${OUT}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});