# Releasing the Brain package

> [!NOTE]
> These steps are to be performed by authorized Voxel51 engineers.

`main` is the trunk: every PR merges to `main`, and nothing originates on a
release branch. `main` always carries the next planned version: cutting a
release branch is immediately followed by a version-bump PR, which always
targets `main`, and that PR is where the next release number is chosen.
Every release is tagged on a `release/vX.Y.Z` branch; tags are never cut
against `main`. Reviewers of version-bump PRs should always check that the
version matches the tag being cut.

## Minor / major release (vX.Y.0)

1. Confirm the `VERSION` file on `main` is `X.Y.0`.

2. Cut `release/vX.Y.0` from `main`. The branch inherits `X.Y.0` from `main`,
   so it needs no version commit.

3. Open a version-bump PR to `main` advancing `VERSION` to the next planned
   version.

4. Publish `vX.Y.0` from `release/vX.Y.0`.

## Patch release (vX.Y.Z)

A patch continues an already released line, so it never touches `main`'s
version.

1. Land the fix on `main` first.

2. Cut `release/vX.Y.Z` from the branch that carried the previous release in
   this line: `release/vX.Y.0` for the first patch, the previous patch's
   branch after that. `release/v0.26.1` branches from `release/v0.26.0`.

3. `git cherry-pick -x` the fix onto the release branch via PR. Release
   branches take cherry-picks only, never a back-merge.

4. Open a PR to `release/vX.Y.Z` setting `VERSION` to `X.Y.Z`.

5. Publish `vX.Y.Z` from `release/vX.Y.Z`.

## Publishing

1. Navigate to the
   [releases page](https://github.com/voxel51/fiftyone-brain/releases) and
   select `Draft a new release`.

2. Select `Create new tag`, enter the tag `vX.Y.Z`, and set the target to the
   release branch.

3. Select `Generate release notes`. Select `Set as the latest release` when
   the tag is the highest version released so far, then `Publish release`.

Pushing the tag triggers the
[build workflow](https://github.com/voxel51/fiftyone-brain/blob/main/.github/workflows/build.yml),
which builds the `.whl` artifacts and publishes them to
[PyPI](https://pypi.org/project/fiftyone-brain/).

## Release candidates

Tag `vX.Y.Z-rc.N` on the release branch. The build workflow checks that the
tag extends the `VERSION` file and builds the rc version from the tag.
