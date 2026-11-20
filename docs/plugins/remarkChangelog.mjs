// Reshape the changelog into release blocks for the Changelog page. Each
// "## [x.y.z] - date" heading and the sections under it become one block: a sticky
// column with the version, date, and jump links, and a column with the sections.
// Other files pass through untouched, and the source stays plain Keep a Changelog.

const attr = (name, value) => ({type: 'mdxJsxAttribute', name, value});
const block = (name, className, children, extra = []) => ({
  type: 'mdxJsxFlowElement',
  name,
  attributes: [attr('className', className), ...extra],
  children,
});
const text = (value) => ({type: 'text', value});
const paragraph = (className, children) => ({
  type: 'paragraph',
  data: {hProperties: {className}},
  children,
});

const plain = (node) =>
  node.value ?? (node.children ?? []).map(plain).join('');

function formatDate(iso) {
  const date = new Date(`${iso}T00:00:00Z`);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

function releaseBlock(heading, body) {
  const label = plain(heading);
  const iso = label.match(/\d{4}-\d{2}-\d{2}/)?.[0];
  const unreleased = /unreleased/i.test(label);

  // Keep only the version in the heading. The date moves to its own line.
  heading.children = heading.children.filter((child) => child.type !== 'text');

  const sections = [];
  for (const node of body) {
    if (node.type === 'heading' && node.depth === 3) {
      sections.push({heading: node, nodes: []});
    } else if (sections.length) {
      sections.at(-1).nodes.push(node);
    }
  }

  const jump = block(
    'nav',
    'changelog-jump',
    sections.map(({heading: h}) =>
      paragraph('changelog-jump-link', [
        {type: 'link', url: `#${h.data?.hProperties?.id ?? ''}`, children: [text(plain(h))]},
      ]),
    ),
  );
  const meta = block(
    'aside',
    'changelog-meta',
    [heading, ...(iso ? [paragraph('changelog-date', [text(formatDate(iso))])] : []), jump],
  );
  const content = block(
    'div',
    'changelog-body',
    sections.length
      ? sections.map(({heading: h, nodes}) =>
          block('div', 'changelog-section', [h, ...nodes], [attr('data-type', plain(h))]),
        )
      : [paragraph('changelog-empty', [text('No changes were recorded for this release.')])],
  );
  return block(
    'section',
    unreleased ? 'changelog-release changelog-unreleased' : 'changelog-release',
    [meta, content],
  );
}

export default function remarkChangelog() {
  return (tree, file) => {
    if (!file.path?.endsWith('/changelog.md')) {
      return;
    }
    const out = [];
    let current = null;
    let filtersAdded = false;
    for (const node of tree.children) {
      if (node.type === 'heading' && node.depth === 2) {
        if (!filtersAdded) {
          out.push({type: 'mdxJsxFlowElement', name: 'ChangelogFilters', attributes: [], children: []});
          filtersAdded = true;
        }
        if (current) out.push(releaseBlock(current.heading, current.body));
        current = {heading: node, body: []};
      } else if (current && node.type === 'definition') {
        out.push(releaseBlock(current.heading, current.body));
        current = null;
        out.push(node);
      } else if (current) {
        current.body.push(node);
      } else {
        out.push(node);
      }
    }
    if (current) out.push(releaseBlock(current.heading, current.body));
    tree.children = out;
  };
}
