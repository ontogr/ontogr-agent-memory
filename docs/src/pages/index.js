import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import InstallCommand from '@site/src/components/InstallCommand';
import styles from './index.module.css';

const INSTALL = 'uv add "ontogr-mem[local,neo4j,postgres]"';

const paths = [
  {
    title: 'Claude Code',
    text: 'Add the memory plugin to Claude Code, then write and recall your first fact.',
    page: 'get-started/claude-code',
  },
  {
    title: 'Other hosts',
    text: 'Connect Codex, Cursor, or OpenCode to the same memory server.',
    page: 'get-started/other-hosts',
  },
  {
    title: 'Python app',
    text: 'Write and search memory from an async application.',
    page: 'get-started/python-app',
  },
  {
    title: 'Shared server and Docker',
    text: 'Run one server for a machine or a container for a team.',
    page: 'get-started/shared-server-and-docker',
  },
  {
    title: 'Terminal chat',
    text: 'Try memory in a terminal before connecting a host.',
    page: 'get-started/chat',
  },
];

export default function Home() {
  const base = useBaseUrl('/');
  const {
    siteConfig: {customFields},
  } = useDocusaurusContext();
  const stablePath = customFields.stablePath ?? 'latest';
  const docsTo = (page) => `${base}${stablePath}/${page}`;

  return (
    <Layout
      title="Durable, traceable memory for AI agents"
      description="OntoGr Agent Memory stores what an agent learned, where it learned it, and when it was true.">
      <header className={styles.hero}>
        <div className="container">
          <h1>OntoGr Agent Memory</h1>
          <p className={styles.pitch}>
            Durable, traceable memory for AI agents. Store what an agent learned,
            where it learned it, and when it was true, then return a budgeted
            context block that an answering model can use.
          </p>
          <div className={styles.actions}>
            <InstallCommand command={INSTALL} />
            <Link className="button button--primary button--lg" to={docsTo('get-started/claude-code')}>
              Get started
            </Link>
          </div>
          <p className={styles.facts}>Python 3.11+ · Apache-2.0</p>
        </div>
      </header>

      <main>
        <section className={styles.section}>
          <div className="container">
            <h2>Pick your path</h2>
            <ul className={styles.paths}>
              {paths.map(({title, text, page}) => (
                <li key={page}>
                  <Link to={docsTo(page)} className={styles.pathLink}>
                    <strong>{title}</strong>
                    <span>{text}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>
    </Layout>
  );
}
