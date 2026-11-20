import React, {useEffect, useRef, useState} from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useDoc, useDocsVersion} from '@docusaurus/plugin-content-docs/client';
import styles from './styles.module.css';

// Icon paths come from Lucide (ISC license) and the simple-icons project (CC0 1.0).
function Icon({children, fill = false}) {
  const props = fill
    ? {fill: 'currentColor'}
    : {fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round'};
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" aria-hidden="true" focusable="false" {...props}>
      {children}
    </svg>
  );
}

const CopyIcon = () => (
  <Icon>
    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
  </Icon>
);
const CheckIcon = () => (
  <Icon>
    <path d="M20 6 9 17l-5-5" />
  </Icon>
);
const ChevronIcon = () => (
  <Icon>
    <path d="m6 9 6 6 6-6" />
  </Icon>
);
const FileIcon = () => (
  <Icon>
    <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
    <path d="M14 2v4a2 2 0 0 0 2 2h4" />
    <path d="M10 9H8" />
    <path d="M16 13H8" />
    <path d="M16 17H8" />
  </Icon>
);
const ChatIcon = () => (
  <Icon>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </Icon>
);
const ClaudeIcon = () => (
  <Icon fill>
    <path d="m4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z" />
  </Icon>
);
const ArrowIcon = () => (
  <svg className={styles.arrow} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
    <path d="M7 7h10v10" />
    <path d="M7 17 17 7" />
  </svg>
);

const askAbout = (url) =>
  encodeURIComponent(`Read from ${url} so I can ask questions about it.`);

/**
 * A "Copy page" button with a menu of ways to hand the page to an AI tool.
 * The Markdown comes from the per-page .md files that docusaurus-plugin-llms writes.
 * Pages without such a file, such as the generated API pages, get no menu.
 */
export default function PageMenu() {
  const {siteConfig} = useDocusaurusContext();
  const {metadata} = useDoc();
  const {version, isLast} = useDocsVersion();
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const root = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const close = (event) => {
      if (event.type === 'keydown' && event.key !== 'Escape') {
        return;
      }
      if (event.type === 'mousedown' && root.current?.contains(event.target)) {
        return;
      }
      setOpen(false);
    };
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', close);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('keydown', close);
    };
  }, [open]);

  if (metadata.id.startsWith('api/')) {
    return null;
  }

  // docusaurus-plugin-llms writes the pages of each version under the path of their source file.
  const outputRoot = version === 'current' ? 'latest' : isLast ? 'stable' : version;
  const markdownPath = `${siteConfig.baseUrl}${outputRoot}/${metadata.source.replace('@site/', '').replace(/\.mdx?$/, '')}.md`;
  const pageUrl = `${siteConfig.url}${metadata.permalink}`;
  const markdownUrl = `${siteConfig.url}${markdownPath}`;

  const copyPage = async () => {
    setOpen(false);
    try {
      const response = await fetch(markdownPath);
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      await navigator.clipboard.writeText(await response.text());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      window.open(markdownPath, '_blank', 'noopener');
    }
  };

  return (
    <div className={styles.root} ref={root}>
      <div className={styles.split}>
        <button type="button" className={styles.copy} onClick={copyPage}>
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied' : 'Copy page'}</span>
        </button>
        <button
          type="button"
          className={clsx(styles.toggle, open && styles.toggleOpen)}
          aria-label="More page options"
          aria-expanded={open}
          onClick={() => setOpen(!open)}>
          <ChevronIcon />
        </button>
      </div>
      {open && (
        <div className={styles.menu}>
          <button type="button" className={styles.item} onClick={copyPage}>
            <span className={styles.tile}><CopyIcon /></span>
            <span className={styles.text}>
              <span className={styles.title}>Copy page</span>
              <span className={styles.hint}>Copy the page as Markdown for LLMs</span>
            </span>
          </button>
          <a className={styles.item} href={markdownPath} target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
            <span className={styles.tile}><FileIcon /></span>
            <span className={styles.text}>
              <span className={styles.title}>View as Markdown<ArrowIcon /></span>
              <span className={styles.hint}>Open the page as plain Markdown</span>
            </span>
          </a>
          <a className={styles.item} href={`https://chat.openai.com/?hints=search&q=${askAbout(pageUrl)}`} target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
            <span className={styles.tile}><ChatIcon /></span>
            <span className={styles.text}>
              <span className={styles.title}>Open in ChatGPT<ArrowIcon /></span>
              <span className={styles.hint}>Ask questions about this page</span>
            </span>
          </a>
          <a className={styles.item} href={`https://claude.ai/new?q=${askAbout(markdownUrl)}`} target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
            <span className={styles.tile}><ClaudeIcon /></span>
            <span className={styles.text}>
              <span className={styles.title}>Open in Claude<ArrowIcon /></span>
              <span className={styles.hint}>Ask questions about this page</span>
            </span>
          </a>
        </div>
      )}
    </div>
  );
}
