import { defineConfig } from 'astro/config';
import { rehypeHeadingIds, unified } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkMermaid } from './src/plugins/remark-mermaid.mjs';
import {
  rehypeFigure,
  rehypeHeadingAnchors,
  rehypeTableScroll,
} from './src/plugins/rehype-content.mjs';
import { SITE } from './src/consts.ts';

export default defineConfig({
  site: SITE.url,
  integrations: [mdx(), sitemap()],
  markdown: {
    processor: unified({
      // remarkMermaid must run before Shiki so mermaid fences escape
      // syntax highlighting and reach the browser as raw diagram source.
      remarkPlugins: [remarkMath, remarkMermaid],
      rehypePlugins: [
        [rehypeKatex, { output: 'html', throwOnError: false }],
        rehypeFigure,
        rehypeTableScroll,
        // Ids first, so the anchors have something to point at.
        rehypeHeadingIds,
        rehypeHeadingAnchors,
      ],
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: false,
    },
  },
});
