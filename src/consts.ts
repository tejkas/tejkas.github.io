// Everything personal lives here. Change these, not the templates.
export const SITE = {
  name: 'Tej Kasturi',
  // Shown under the name on the home page.
  tagline: "I'm documenting my projects, learning notes, and thoughts here.",
  // Used for RSS, sitemap, and <meta> tags. Must match where the site is
  // actually served, or canonical links and feed URLs point at the wrong host.
  url: 'https://tejkas.github.io',
  description: 'Technical project writeups, learning notes, and thoughts.',
  // How many posts to show on the home page.
  recentPostCount: 5,
};

// Shown on the home page and in the footer.
export const LINKS = [
  { label: 'GitHub', href: 'https://github.com/tejkas' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/tejasvi-kasturi/' },
  { label: 'RSS', href: '/rss.xml' },
];

// Kept off the home page and footer so it isn't scraped from every page;
// it appears only under Contact on the About page.
export const EMAIL = 'kasturitejasvi@gmail.com';

export const NAV = [
  { label: 'Blog', href: '/blog' },
  { label: 'About', href: '/about' },
];
