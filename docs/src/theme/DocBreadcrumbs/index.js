import React from 'react';
import DocBreadcrumbs from '@theme-original/DocBreadcrumbs';
import PageMenu from '@site/src/components/PageMenu';
import styles from './styles.module.css';

// The page menu shares a row with the breadcrumbs, so a long page title never wraps around it.
export default function DocBreadcrumbsWrapper(props) {
  return (
    <div className={styles.row}>
      <div className={styles.crumbs}>
        <DocBreadcrumbs {...props} />
      </div>
      <PageMenu />
    </div>
  );
}
