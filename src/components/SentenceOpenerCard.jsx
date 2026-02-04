import React from 'react';
import './SentenceOpenerCard.css';

const SentenceOpenerCard = ({ sentenceOpenerData }) => {
    if (!sentenceOpenerData) {
        return null;
    }

    const {
        status,
        message,
        openers_found = {},
        total_sentences = 0,
        recommendations = [],
        variety_score = 100
    } = sentenceOpenerData;

    // Determine status styling
    const getStatusClass = () => {
        if (status === 'excellent') return 'status-excellent';
        if (status === 'good') return 'status-good';
        if (status === 'needs_improvement') return 'status-warning';
        return 'status-neutral';
    };

    const getScoreColor = (score) => {
        if (score >= 80) return '#4caf50';
        if (score >= 60) return '#ff9800';
        return '#f44336';
    };

    return (
        <div className="sentence-opener-card">
            <div className="card-header">
                <h3>📝 Sentence Openers</h3>
                <div className="variety-score" style={{ color: getScoreColor(variety_score) }}>
                    Variety Score: {variety_score}/100
                </div>
            </div>

            <div className={`status-message ${getStatusClass()}`}>
                {message}
            </div>

            {total_sentences > 0 && (
                <div className="sentence-count">
                    Total Sentences Analyzed: <strong>{total_sentences}</strong>
                </div>
            )}

            {Object.keys(openers_found).length > 0 && (
                <div className="openers-section">
                    <h4>Detected Sentence Openers:</h4>
                    <div className="openers-grid">
                        {Object.entries(openers_found).map(([opener, data]) => (
                            <div
                                key={opener}
                                className={`opener-item ${data.is_overused ? 'overused' : ''} severity-${data.severity}`}
                            >
                                <div className="opener-header">
                                    <span className="opener-word">"{opener.toUpperCase()}"</span>
                                    <span className="opener-count">{data.count}x</span>
                                </div>
                                <div className="opener-percentage">
                                    {data.percentage}% of sentences
                                </div>
                                {data.is_overused && (
                                    <div className="opener-warning">
                                        ⚠️ Overused
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {recommendations.length > 0 && (
                <div className="recommendations-section">
                    <h4>💡 Recommendations:</h4>
                    {recommendations.map((rec, index) => (
                        <div
                            key={index}
                            className={`recommendation-item severity-${rec.severity}`}
                        >
                            <div className="rec-header">
                                <span className="rec-opener">{rec.opener}</span>
                                <span className="rec-usage">{rec.usage}</span>
                            </div>
                            <div className="rec-impact">
                                <strong>Impact:</strong> {rec.impact}
                            </div>
                            <div className="rec-suggestion">
                                <strong>Try instead:</strong> {rec.suggestion}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {status === 'excellent' && (
                <div className="excellent-feedback">
                    <div className="feedback-icon">✨</div>
                    <div className="feedback-text">
                        Excellent sentence variety! Your speech demonstrates good use of
                        varied openers, which enhances engagement and flow.
                    </div>
                </div>
            )}
        </div>
    );
};

export default SentenceOpenerCard;
