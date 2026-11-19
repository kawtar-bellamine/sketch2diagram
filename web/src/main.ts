import mermaid from "mermaid";
import "./style.css";

interface Result { mermaid: string; issues: string[]; }

mermaid.initialize({ startOnLoad: false, theme: "neutral", flowchart: { curve: "basis" } });

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const desc = $<HTMLTextAreaElement>("desc");
const code = $<HTMLTextAreaElement>("code");
const issues = $<HTMLDivElement>("issues");
const diagram = $<HTMLDivElement>("diagram");

desc.value = "The process starts when a part arrives at the inspection station. First, the inspector measures the " +
  "dimensions. If the part is within tolerance, the process continues; otherwise, the part is reworked and the " +
  "dimensions are measured again. Then the result is recorded. The process then ends.";

let n = 0;
async function draw(src: string): Promise<void> {
  try {
    const { svg } = await mermaid.render(`g${n++}`, src);
    diagram.innerHTML = svg;
  } catch (e) {
    diagram.innerHTML = `<div class="err">Mermaid syntax error: ${(e as Error).message}</div>`;
  }
}

function show(r: Result): void {
  code.value = r.mermaid;
  issues.innerHTML = r.issues.length
    ? `<b>To check:</b><ul>${r.issues.map((i) => `<li>${i}</li>`).join("")}</ul>`
    : `<span class="ok">Structure valid: one start, every decision has two branches, every step is reachable.</span>`;
  void draw(r.mermaid);
}

$<HTMLButtonElement>("go-text").onclick = async () => {
  const r = await fetch("/api/text", { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify({ description: desc.value }) });
  show(await r.json());
};

$<HTMLInputElement>("file").onchange = async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  $<HTMLImageElement>("preview").src = URL.createObjectURL(f);
  const body = new FormData(); body.append("file", f);
  show(await (await fetch("/api/sketch", { method: "POST", body })).json());
};

code.oninput = () => void draw(code.value);

for (const [tab, on, off] of [["t-text", "p-text", "p-sketch"], ["t-sketch", "p-sketch", "p-text"]]) {
  $<HTMLButtonElement>(tab).onclick = () => {
    $(on).hidden = false; $(off).hidden = true;
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.id === tab));
  };
}
$<HTMLButtonElement>("go-text").click();
