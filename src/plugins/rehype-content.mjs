import { visit } from 'unist-util-visit';

/**
 * Promotes a standalone image to a <figure>, using the Markdown title as
 * the caption:  ![alt text](./chart.png "The caption")
 *
 * Markdown has no figure syntax, so an image otherwise lands in a bare
 * <p> with its title attribute showing only as a browser tooltip — which
 * is invisible on touch devices and to screen readers.
 *
 * Alt and caption stay distinct on purpose: alt describes the image for
 * someone who can't see it, the caption adds context for everyone.
 */
export function rehypeFigure() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'p' || !parent) return;

      const meaningful = node.children.filter(
        (c) => !(c.type === 'text' && c.value.trim() === ''),
      );
      if (meaningful.length !== 1) return;

      const img = meaningful[0];
      if (img.type !== 'element' || img.tagName !== 'img') return;

      const caption = img.properties?.title;
      // Title doubles as the tooltip; drop it now that it's a caption.
      if (caption) delete img.properties.title;

      parent.children[index] = {
        type: 'element',
        tagName: 'figure',
        properties: {},
        children: [
          img,
          ...(caption
            ? [
                {
                  type: 'element',
                  tagName: 'figcaption',
                  properties: {},
                  children: [{ type: 'text', value: caption }],
                },
              ]
            : []),
        ],
      };
    });
  };
}

/**
 * Wraps tables in a scrollable div.
 *
 * A results table with five columns overflows a phone screen. Without a
 * wrapper the whole page scrolls sideways, which breaks every other line
 * of text too; with one, only the table moves.
 */
export function rehypeTableScroll() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'table' || !parent) return;
      if (parent.type === 'element' && parent.properties?.className?.includes?.('table-scroll')) {
        return;
      }

      parent.children[index] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-scroll'], tabIndex: 0, role: 'region' },
        children: [node],
      };
    });
  };
}
