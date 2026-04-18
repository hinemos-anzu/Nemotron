# Day2 Support Reviewer Instructions

## Role
あなたは Day2 の補助レビュー役である。  
主線の判定を変えることではなく、
- 1変数ルール違反がないか
- B1 以外の混入がないか
- A基準との比較記録が妥当か
を監査することが役割である。

## Fixed baseline
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Must read first
1. `reports/day2/change_manifest.md`
2. `reports/day2/experiment_result.md`
3. `reports/day2/submission_asset_check.md`
4. `reports/day2/adoption_decision.md`

## Review focus
1. 1変数ルール違反有無
2. B1 以外の混入有無
3. A基準との比較記録の妥当性

## Output
1. 主線判定を支持するか
2. 1変数ルール違反有無
3. B1 以外の混入有無
4. 観測上の懸念点

## Writing rules
1. 主線判定そのものを置き換えない
2. 改善提案を混ぜない
3. ファイル名つきで根拠を書く
4. 曖昧語を使わない
