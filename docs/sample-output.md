# Sample output

This example uses the fictional records in `sample-claims-data.csv` and local TF-IDF mode. It is generated from the included data, not a production result.

## Portfolio snapshot

| Measure | Result |
| --- | ---: |
| Claims | 8 |
| Open | 3 |
| Pending | 2 |
| Closed | 3 |
| Total incurred | $41,080.00 |
| Average incurred | $5,135.00 |
| Median incurred | $3,700.00 |
| Total reserve | $14,010.00 |
| Average days open | 56.8 |

## Example question

> Review claim CLM-10001 and find collision coverage guidance.

## Example response excerpt

> **Claim record:** CLM-10001 is a Collision Auto claim with status Open, incurred loss $4,200.00, reserve $1,900.00, and 42 days open.
>
> **Historical comparison:** 2 Collision records; median incurred loss $3,700.00 and median days open 52. This is descriptive context, not a claim forecast.
>
> **Policy research lead:** [POL-AUTO-01] Personal auto collision coverage. The synthetic summary says collision coverage may apply to direct accidental damage from impact, subject to the declarations, deductible, exclusions, and policy period.
>
> **Human review:** Confirm the applicable contract wording and claim facts before making any decision.

This excerpt is abbreviated. The app displays the full retrieved synthetic passage and notes that it is not contract wording.

