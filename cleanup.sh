#!/usr/bin/env bash
# Trim the ml-agents fork down to: the Soccer environment + the Python training stack.
# Run from the ROOT of your fork, on a branch, e.g.:
#   git switch -c chore/strip-repo && bash cleanup-fork.sh && git status
set -euo pipefail

EX="Project/Assets/ML-Agents/Examples"

# --- 1. Unity example environments (keep Soccer + SharedAssets) -------------
for d in 3DBall Basic Crawler DungeonEscape FoodCollector GridWorld Hallway \
         Match3 PushBlock PushBlockWithInput Pyramids Sorter Startup Walker \
         WallJump Worm; do
  git rm -rq --ignore-unmatch "$EX/$d" "$EX/$d.meta"
done
git rm -q --ignore-unmatch "$EX/GridFoodCollector.onnx" "$EX/GridFoodCollector.onnx.meta"

# --- 2. Unity extras we don't need -----------------------------------------
git rm -rq --ignore-unmatch Project/Assets/ML-Agents/TestScenes Project/Assets/ML-Agents/TestScenes.meta
git rm -rq --ignore-unmatch Project/Recordings
git rm -rq --ignore-unmatch DevProject
git rm -rq --ignore-unmatch com.unity.ml-agents.extensions   # see note: edit Packages/manifest.json after

# --- 3. Training configs (keep the two POCA soccer configs) ----------------
git rm -rq --ignore-unmatch config/imitation config/ppo config/sac
git rm -q  --ignore-unmatch config/poca/DungeonEscape.yaml config/poca/PushBlockCollab.yaml

# --- 4. Upstream docs / CI / infra -----------------------------------------
git rm -rq --ignore-unmatch docs localized_docs mkdocs.yml
git rm -rq --ignore-unmatch colab colab_requirements.txt
git rm -rq --ignore-unmatch .yamato ml-agents/tests/yamato
git rm -rq --ignore-unmatch .github/workflows .github/ISSUE_TEMPLATE .github/PULL_REQUEST_TEMPLATE.md .github/stale.yml
git rm -rq --ignore-unmatch Dockerfile unity-volume SURVEY.md
git rm -q  --ignore-unmatch markdown-link-check.fast.json markdown-link-check.full.json
git rm -rq --ignore-unmatch protobuf-definitions
git rm -q  --ignore-unmatch utils/generate_markdown_docs.py utils/make_readme_table.py \
                            utils/run_markdown_link_check.py utils/validate_release_links.py

echo
echo "Done. Remaining manual steps:"
echo "  1. Edit Project/Packages/manifest.json — remove the"
echo "     'com.unity.ml-agents.extensions' dependency line AND its 'testables' entry."
echo "  2. Replace README.md with your own project README."
echo "  3. Open the project in Unity and load Examples/Soccer/Scenes/SoccerTwos.unity."
