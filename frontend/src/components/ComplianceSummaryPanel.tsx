import type { ComplianceSummary } from "../types/analysis";

export function ComplianceSummaryPanel({ compliance }: { compliance: ComplianceSummary }) {
  const cards = [
    { label: "Shelves scanned", value: compliance.total_regions },
    { label: "Compliant", value: compliance.ok_regions },
    { label: "Under-stocked", value: compliance.understocked_regions },
    { label: "Empty", value: compliance.empty_regions },
    { label: "Total facings", value: compliance.total_facings },
    { label: "Est. missing facings", value: compliance.total_estimated_missing_facings },
  ];

  return (
    <div className="compliance-panel">
      <div className="compliance-panel__headline">
        <span className="compliance-panel__ratio">
          {(compliance.overall_occupancy_ratio * 100).toFixed(0)}%
        </span>
        <span>overall shelf occupancy</span>
      </div>
      <div className="compliance-panel__grid">
        {cards.map((card) => (
          <div className="stat-card" key={card.label}>
            <span className="stat-card__value">{card.value}</span>
            <span className="stat-card__label">{card.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
