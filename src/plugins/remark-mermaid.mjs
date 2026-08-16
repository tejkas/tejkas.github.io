import { visit } from 'unist-util-visit';

/**
 * Turns ```mermaid fences into <div class="mermaid"> before Shiki sees them.
 *
 * Without this, Shiki treats the diagram source as an unknown language and
 * renders it as a plain code block. Mermaid then never gets a chance to draw
 * it, because the text is buried in <span>s rather than sitting as the raw
 * textContent its parser expects.
 */
export function remarkMermaid() {
  return (tree) => {
    visit(tree, 'code', (node, index, parent) => {
      if (node.lang !== 'mermaid' || !parent) return;
      parent.children[index] = {
        type: 'html',
        value: `<div class="mermaid-wrap"><pre class="mermaid">${escapeHtml(
          node.value,
        )}</pre></div>`,
      };
    });
  };
}

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
