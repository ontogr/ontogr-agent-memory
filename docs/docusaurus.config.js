// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// See: https://docusaurus.io/docs/api/docusaurus-config

import {readFileSync} from 'node:fs';
import {themes as prismThemes} from 'prism-react-renderer';
import {iconLinkHtml} from './navbarIcons.js';
import remarkChangelog from './plugins/remarkChangelog.mjs';
import {syncChangelog} from './plugins/syncChangelog.mjs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

syncChangelog();

// Preview deploys override the base URL with the BASE_URL env var (e.g. BASE_URL=/).
const baseUrl = process.env.BASE_URL ?? '/ontogr-agent-memory/';

// Release versions, newest first. The site has no frozen release until the
// release pull request runs `make docs-version`.
const releases = JSON.parse(readFileSync(new URL('./versions.json', import.meta.url), 'utf8'));
const stablePath = releases.length > 0 ? 'stable' : 'latest';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'OntoGr Agent Memory',
  tagline: 'Durable, traceable memory for AI agents.',

  future: {
    v4: true,
  },

  // Served at https://ontogr.github.io/ontogr-agent-memory/
  url: 'https://ontogr.github.io',
  baseUrl,

  organizationName: 'ontogr',
  projectName: 'ontogr-agent-memory',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  markdown: {
    mermaid: true,
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkChangelog],
          // The newest release is served at /stable/ and the docs of the main
          // branch at /latest/. A release is a frozen copy of the docs added
          // by `make docs-version`. Until the first release, there is no
          // frozen version and the version dropdown is omitted.
          versions: {
            current: {label: 'latest', path: 'latest', banner: 'unreleased'},
            ...(releases.length > 0
              ? {[releases[0]]: {label: `${releases[0]} (stable)`, path: 'stable'}}
              : {}),
          },
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        docsRouteBasePath: '/',
        indexBlog: false,
        language: ['en'],
      }),
    ],
  ],

  plugins: [
    // Return the text of an imported file with `?raw`, so the host snippets
    // frozen with each version stay beside the page that shows them.
    function rawSnippetsPlugin() {
      return {
        name: 'raw-snippets',
        configureWebpack() {
          return {
            module: {
              rules: [{resourceQuery: /raw/, type: 'asset/source'}],
            },
          };
        },
      };
    },
    [
      'docusaurus-plugin-llms',
      {
        title: 'OntoGr Agent Memory',
        description:
          'Durable, traceable memory for AI agents. Store what an agent learned, where it learned it, and when it was true.',
        ignoreFiles: ['api/**'],
        generateMarkdownFiles: true,
        versions: [
          {
            name: 'current',
            label: 'latest',
            docsDir: 'docs',
            path: 'latest',
            routePrefix: 'latest',
          },
          ...releases.map((id, index) => ({
            name: id,
            docsDir: `versioned_docs/version-${id}`,
            path: index === 0 ? 'stable' : id,
            routePrefix: index === 0 ? 'stable' : id,
          })),
        ],
      },
    ],
  ],

  customFields: {
    /** The path of the newest published release, or `latest` before one exists. */
    stablePath,
  },

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      mermaid: {
        theme: {light: 'neutral', dark: 'dark'},
      },
      navbar: {
        title: 'OntoGr Agent Memory',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
            'data-text': 'Documentation',
          },
          {
            type: 'docSidebar',
            sidebarId: 'referenceSidebar',
            position: 'left',
            label: 'Reference',
            'data-text': 'Reference',
          },
          {
            type: 'doc',
            docId: 'changelog',
            position: 'left',
            label: 'Changelog',
            'data-text': 'Changelog',
          },
          {type: 'docsVersionDropdown', position: 'right'},
          {type: 'search', position: 'right'},
          {
            href: 'https://github.com/ontogr/ontogr-agent-memory',
            html: iconLinkHtml('github', 'GitHub'),
            'aria-label': 'GitHub',
            className: 'navbar-icon',
            position: 'right',
          },
          {
            href: 'https://pypi.org/project/ontogr-mem/',
            html: iconLinkHtml('pypi', 'PyPI'),
            'aria-label': 'PyPI',
            className: 'navbar-icon navbar-icon--pypi',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'light',
        links: [
          {
            title: 'Get started',
            items: [
              {label: 'Claude Code', to: `/${stablePath}/get-started/claude-code`},
              {label: 'Other hosts', to: `/${stablePath}/get-started/other-hosts`},
              {label: 'Python app', to: `/${stablePath}/get-started/python-app`},
              {label: 'Shared server and Docker', to: `/${stablePath}/get-started/shared-server-and-docker`},
              {label: 'Terminal chat', to: `/${stablePath}/get-started/chat`},
            ],
          },
          {
            title: 'Learn',
            items: [
              {label: 'How data flows', to: `/${stablePath}/concepts/how-data-flows`},
              {label: 'Facts and provenance', to: `/${stablePath}/concepts/facts-and-provenance`},
              {label: 'Configuration', to: `/${stablePath}/reference/configuration`},
              {label: 'Python API', to: `/${stablePath}/api/`},
              {label: 'Changelog', to: `/${stablePath}/changelog`},
            ],
          },
          {
            title: 'Community',
            items: [
              {label: 'GitHub', href: 'https://github.com/ontogr/ontogr-agent-memory'},
              {label: 'PyPI', href: 'https://pypi.org/project/ontogr-mem/'},
              {
                label: 'Issues',
                href: 'https://github.com/ontogr/ontogr-agent-memory/issues',
              },
              {
                label: 'Contributing',
                href: 'https://github.com/ontogr/ontogr-agent-memory/blob/main/CONTRIBUTING.md',
              },
            ],
          },
          {
            title: 'Legal',
            items: [
              {
                label: 'License (Apache-2.0)',
                href: 'https://github.com/ontogr/ontogr-agent-memory/blob/main/LICENSE',
              },
              {
                label: 'Security policy',
                href: 'https://github.com/ontogr/ontogr-agent-memory/blob/main/SECURITY.md',
              },
              {
                label: 'Code of conduct',
                href: 'https://github.com/ontogr/ontogr-agent-memory/blob/main/CODE_OF_CONDUCT.md',
              },
            ],
          },
        ],
        copyright: `ontogr-mem is open source under the Apache-2.0 license. Copyright &copy; ${new Date().getFullYear()} the OntoGr Agent Memory contributors.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
