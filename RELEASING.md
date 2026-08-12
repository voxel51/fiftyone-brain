# Releasing the Brain package

> [!NOTE]
> These steps are to be performed by authorized Voxel51 engineers.

`main` is the trunk: every PR merges to `main`, and nothing originates on a
release branch. Between releases, `VERSION` in `setup.py` is the next planned
version. Reviewers of version-bump PRs should always check that the version
matches the tag being cut.

## Minor / major release (vX.Y.0)

1. Confirm `VERSION` in `setup.py` on `main` is `X.Y.0`.

1. Navigate to the
   [releases page](https://github.com/voxel51/fiftyone-brain/releases) and
   select `Draft a new release`.

1. Select `Create new tag` with tag `vX.Y.0` and set the target to `main`.

1. Select `Generate release notes`, then `Set as the latest release`, then
   `Publish release`.

   Pushing the tag triggers the
   [build workflow](https://github.com/voxel51/fiftyone-brain/blob/main/.github/workflows/build.yml),
   which builds the `.whl` artifacts and publishes them to
   [PyPI](https://pypi.org/project/fiftyone-brain/).

1. Open a version-bump PR to `main` advancing `VERSION` to the next planned
   version.

## Patch release (vX.Y.Z)

1. Cut `release/vX.Y.Z` from the `vX.Y.Z-1` tag.

1. Land the fixes on `main` first, then `git cherry-pick -x` them to the
   release branch via PR. A release branch accepts only cherry-picks of
   `main` commits and its version bump — no back-merges in either direction.

1. Open a PR to the release branch bumping `VERSION` to `X.Y.Z`.

1. Draft the GitHub release with tag `vX.Y.Z` targeting `release/vX.Y.Z`,
   following the steps above.

## Release candidates

Tag `vX.Y.Z-rc.N` on the branch being released. The build workflow builds
the rc version via the `RELEASE_VERSION` environment variable and validates
it against `setup.py`.
