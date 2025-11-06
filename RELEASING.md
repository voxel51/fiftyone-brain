# Releasing the Brain package

> [!NOTE]
> These steps are to be performed by authorized Voxel51 engineers.

The `fiftyone-brain` repository follows `Gitflow`.
Releases will be initiated when a teammate submits a 
pull request from their respective `release/v*` branch to `main`.
We can see an example PR for
[version 0.21.4](https://github.com/voxel51/fiftyone-brain/pull/265). 
Reviewers should always check that the version in the `setup.py`
matches the branch version.

The release engineer will merge the pull request once it is approved.

The PyPI uploads will be triggered when a release tag is pushed to the
repository:

1. Navigate to the
   [releases page](https://github.com/voxel51/fiftyone-brain/pull/265).

1. Select `Draft a new release`.

1. Select `Create new tag` with the appropriate version and set the target to
   `main`.

    1. The tag format is `v<semantic-version>`.
       For example, `v0.21.4`. 
       This should match the `setup.py` and release branch.

1. Select `Generate release notes`.

1. Select `Set as the latest release`.

1. Select `Publish release`.

This will create a new tag in the repository and will trigger the
[build/publish workflow](https://github.com/voxel51/fiftyone-brain/actions/workflows/build.yml).
This workflow will build the `.whl` artifacts and publish them to
[PyPI](https://pypi.org/project/fiftyone-brain/).

Once the build are finished, submit a PR from `main` to `develop` to complete
the `Gitflow` process.