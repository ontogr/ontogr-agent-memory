import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import {
  useActivePlugin,
  useDocVersionSuggestions,
  useDocsPreferredVersion,
  useDocsVersion,
} from '@docusaurus/plugin-content-docs/client';
import {ThemeClassNames} from '@docusaurus/theme-common';

// The suggested version link is labelled with the version name (e.g. "0.0.6"),
// not its label ("0.0.6 (stable)"), so the text stays readable.

function UnreleasedVersionLabel() {
  return (
    <Translate
      id="theme.docs.versions.unreleasedVersionLabel.main"
      description="Tells the user they are browsing docs for the unreleased version on main">
      This is documentation for the unreleased version on main.
    </Translate>
  );
}

function UnmaintainedVersionLabel({siteTitle, versionMetadata}) {
  return (
    <Translate
      id="theme.docs.versions.unmaintainedVersionLabel"
      description="The label used to tell the user that he's browsing an unmaintained doc version"
      values={{
        siteTitle,
        versionLabel: <b>{versionMetadata.label}</b>,
      }}>
      {
        'This is documentation for {siteTitle} {versionLabel}, which is no longer actively maintained.'
      }
    </Translate>
  );
}

const BannerLabelComponents = {
  unreleased: UnreleasedVersionLabel,
  unmaintained: UnmaintainedVersionLabel,
};

function BannerLabel(props) {
  const BannerLabelComponent =
    BannerLabelComponents[props.versionMetadata.banner];
  return <BannerLabelComponent {...props} />;
}

function UnreleasedSuggestionLabel({versionName, to, onClick}) {
  return (
    <Translate
      id="theme.docs.versions.latestReleaseSuggestionLabel"
      description="Tells the user where to find up-to-date documentation for the latest release"
      values={{
        latestVersionLink: (
          <b>
            <Link to={to} onClick={onClick}>
              {`${versionName} version documentation`}
            </Link>
          </b>
        ),
      }}>
      {
        'For up-to-date documentation of the latest release, see the {latestVersionLink}.'
      }
    </Translate>
  );
}

function LatestVersionSuggestionLabel({versionLabel, to, onClick}) {
  return (
    <Translate
      id="theme.docs.versions.latestVersionSuggestionLabel"
      description="The label used to tell the user to check the latest version"
      values={{
        versionLabel,
        latestVersionLink: (
          <b>
            <Link to={to} onClick={onClick}>
              <Translate
                id="theme.docs.versions.latestVersionLinkLabel"
                description="The label used for the latest version suggestion link label">
                latest version
              </Translate>
            </Link>
          </b>
        ),
      }}>
      {
        'For up-to-date documentation, see the {latestVersionLink} ({versionLabel}).'
      }
    </Translate>
  );
}

function DocVersionBannerEnabled({className, versionMetadata}) {
  const {
    siteConfig: {title: siteTitle},
  } = useDocusaurusContext();
  const {pluginId} = useActivePlugin({failfast: true});
  const getVersionMainDoc = (version) =>
    version.docs.find((doc) => doc.id === version.mainDocId);
  const {savePreferredVersionName} = useDocsPreferredVersion(pluginId);
  const {latestDocSuggestion, latestVersionSuggestion} =
    useDocVersionSuggestions(pluginId);
  // Link to the same doc in the suggested version when possible, otherwise its main doc.
  const latestVersionSuggestedDoc =
    latestDocSuggestion ?? getVersionMainDoc(latestVersionSuggestion);
  return (
    <div
      className={clsx(
        className,
        ThemeClassNames.docs.docVersionBanner,
        'alert alert--warning margin-bottom--md',
      )}
      role="alert">
      <div>
        <BannerLabel siteTitle={siteTitle} versionMetadata={versionMetadata} />
      </div>
      <div className="margin-top--md">
        {versionMetadata.banner === 'unreleased' ? (
          <UnreleasedSuggestionLabel
            versionName={latestVersionSuggestion.name}
            to={latestVersionSuggestedDoc.path}
            onClick={() => savePreferredVersionName(latestVersionSuggestion.name)}
          />
        ) : (
          <LatestVersionSuggestionLabel
            versionLabel={latestVersionSuggestion.label}
            to={latestVersionSuggestedDoc.path}
            onClick={() => savePreferredVersionName(latestVersionSuggestion.name)}
          />
        )}
      </div>
    </div>
  );
}

export default function DocVersionBanner({className}) {
  const versionMetadata = useDocsVersion();
  if (versionMetadata.banner) {
    return (
      <DocVersionBannerEnabled
        className={className}
        versionMetadata={versionMetadata}
      />
    );
  }
  return null;
}
