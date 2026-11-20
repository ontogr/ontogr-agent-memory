import {useState} from 'react';
import styles from './styles.module.css';

function copyText(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // navigator.clipboard is missing on pages served over plain HTTP.
  const field = document.createElement('textarea');
  field.value = text;
  document.body.appendChild(field);
  field.select();
  document.execCommand('copy');
  field.remove();
  return Promise.resolve();
}

/** A one-line shell command with a copy button. */
export default function InstallCommand({command, className}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await copyText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className={`${styles.command} ${className ?? ''}`}>
      <span className={styles.prompt} aria-hidden="true">$</span>
      <code>
        {command.split(' ').map((word, i) => (
          <span key={i}>
            {i > 0 && ' '}
            {word.startsWith('--') ? <span className={styles.flag}>{word}</span> : word}
          </span>
        ))}
      </code>
      <button type="button" className={styles.copy} onClick={copy}>
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
}
