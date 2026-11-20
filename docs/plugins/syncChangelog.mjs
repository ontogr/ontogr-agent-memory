import {existsSync, readFileSync, writeFileSync} from 'node:fs';

const source = new URL('../../CHANGELOG.md', import.meta.url);
const target = new URL('../docs/changelog.md', import.meta.url);

/**
 * Copy the root CHANGELOG.md into docs/changelog.md, which is git-ignored.
 * The docs plugin builds only the files under docs/, and a symlink does not
 * work because the bundler follows it to the real path outside that folder.
 * The site config calls this, so both start and build pick up the latest file.
 */
export function syncChangelog() {
  const text = readFileSync(source, 'utf8');
  if (!existsSync(target) || readFileSync(target, 'utf8') !== text) {
    writeFileSync(target, text);
  }
}
