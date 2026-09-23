# Releasing the Brain package

> [!NOTE]
> These steps are to be performed by authorized Voxel51 engineers.

`main` is the trunk: changes land on `main` first, and a release branch takes
`git cherry-pick -x` of commits already on `main`. The rare fix that applies
only to a release branch goes straight to it with the `release-only-fix`
label, and its PR body says why `main` doesn't need it.

`VERSION` always holds the exact version being built, and a tag must match it
exactly. `main` carries the next planned version as `X.Y.0.devN`; a release
branch carries `X.Y.ZrcN` until it is finalized to `X.Y.Z`. Every suffix
bump is automated:

- Creating `release/vX.Y.Z` commits `X.Y.Zrc0` to it, or the next unused
  rc number if one is already tagged. For a minor or major branch (`Z` is
  `0`), it also opens a version-bump PR to `main` setting `X.Y+1.0.dev0`.
  Edit that PR to choose a different next version.
- Publishing `vX.Y.0.devN` from `main` commits `X.Y.0.devN+1` to `main`.
- Publishing `vX.Y.ZrcN` commits `X.Y.ZrcN+1` to `release/vX.Y.Z`.

## Minor / major release (vX.Y.0)

1. Cut `release/vX.Y.0` from `main`.

2. Publish release candidates from `release/vX.Y.0` until one is ready.

3. [Finalize](#finalizing) `release/vX.Y.0` and publish `vX.Y.0`.

## Patch release (vX.Y.Z)

A patch continues an already released line, so it never touches `main`'s
version.

1. Land the fix on `main` first.

2. Cut `release/vX.Y.Z` from the branch that carried the previous release in
   this line: `release/vX.Y.0` for the first patch, the previous patch's
   branch after that. `release/v0.26.1` branches from `release/v0.26.0`.

3. Cherry-pick the fix onto the release branch via PR. Label the `main` PR
   `cherry-pick-to-release` and the release PR opens when it merges, or run
   the
   [Cherry-Pick to Release workflow](https://github.com/voxel51/fiftyone-brain/actions/workflows/cherry-pick.yml)
   for a PR that already merged. Release branches take cherry-picks only,
   never a back-merge.

4. [Finalize](#finalizing) `release/vX.Y.Z` and publish `vX.Y.Z`.

## Finalizing

Run the
[Release Branch workflow](https://github.com/voxel51/fiftyone-brain/actions/workflows/release-branch.yml)
with the release branch and `finalize` checked. It commits the bare `X.Y.Z`
to `VERSION`, and the branch is ready to publish.

## Publishing (Aloha only)

1. Navigate to the
   [releases page](https://github.com/voxel51/fiftyone-brain/releases) and
   select `Draft a new release`.

2. Select `Create new tag`, enter the tag matching the branch's `VERSION`
   with a `v` prefix, and set the target to that branch.

3. Select `Generate release notes`. For an `rcN` or `.devN` tag, select
   `Set as a pre-release`. Otherwise select `Set as the latest release` when
   the tag is the highest version released so far. Then `Publish release`.

Pushing the tag triggers the
[build workflow](https://github.com/voxel51/fiftyone-brain/blob/main/.github/workflows/build.yml),
which builds the `.whl` artifacts and publishes them to
[PyPI](https://pypi.org/project/fiftyone-brain/).

## Release candidates and dev builds

Publish the tag that matches the branch's current `VERSION`: `vX.Y.ZrcN` on
a release branch, `vX.Y.0.devN` on `main`. The next number is committed once
the build publishes.
