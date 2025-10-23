import React from 'react';
import { TradeExplanation } from '../../types';
import { formatTimestamp } from '../../utils/formatters';
import { ConfidenceScore, SupportingDataVisualization } from './index';

interface ExplanationCardProps {
    explanation: TradeExplanation;
    className?: string;
    showSupportingData?: boolean;
    compact?: boolean;
}

export function ExplanationCard({
    explanation,
    className = '',
    showSupportingData = true,
    compact = false
}: ExplanationCardProps) {
    return (
        <div className={`bg-white rounded-lg border shadow-sm ${className}`}>
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                    <h3 className={`font-medium text-gray-900 ${compact ? 'text-sm' : 'text-base'}`}>
                        Trade Explanation
                    </h3>
                    <div className="flex items-center space-x-3">
                        <ConfidenceScore
                            confidence={explanation.confidence}
                            size={compact ? 'sm' : 'md'}
                        />
                        <span className="text-xs text-gray-500">
                            {formatTimestamp(explanation.timestamp, true)}
                        </span>
                    </div>
                </div>
            </div>

            {/* Explanation Content */}
            <div className="p-4">
                <div className={`text-gray-700 leading-relaxed ${compact ? 'text-sm' : 'text-base'}`}>
                    {explanation.explanation}
                </div>
            </div>

            {/* Supporting Data */}
            {showSupportingData && explanation.supporting_data &&
                Object.keys(explanation.supporting_data).length > 0 && (
                    <div className="border-t border-gray-200">
                        <SupportingDataVisualization
                            data={explanation.supporting_data}
                            compact={compact}
                        />
                    </div>
                )}
        </div>
    );
}

export default ExplanationCard;