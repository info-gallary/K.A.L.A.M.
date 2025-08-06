// components/ReportHistorySection.jsx
import React from 'react';
import { FileText, Download, Trash2 } from 'lucide-react';

const ReportHistorySection = ({
  reportHistory,
  deleteReport,
  themeClasses
}) => {
  return (
    <div className="flex-1 p-4 overflow-y-auto">
      <h2 className={`text-base font-semibold ${themeClasses.text} mb-3`}>Report History</h2>
      
      <div className={`${themeClasses.card} border rounded`}>
        <div className={`px-4 py-2 border-b ${themeClasses.border}`}>
          <h3 className={`text-sm font-medium ${themeClasses.text}`}>Recent Reports</h3>
        </div>
        
        <div className={`divide-y ${themeClasses.border}`}>
          
          {reportHistory.map((report) => (
            <div key={report.id} className={`px-4 py-3 ${themeClasses.hover} transition-colors`}>
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h4 className={`text-sm font-medium ${themeClasses.text}`}>{report.name}</h4>
                  <div className={`flex items-center text-xs ${themeClasses.textMuted} mt-0.5 flex-wrap gap-1`}>
                    <span>Date: {report.date}</span>
                    <span>•</span>
                    <span>Time: {report.time}</span>
                    <span>•</span>
                    <span>Created: {report.createdAt}</span>
                    <span>•</span>
                    <span>Source: {report.source}</span>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    report.status === 'Completed' 
                      ? `${themeClasses.success} ${themeClasses.successBg} border ${themeClasses.successBorder}`
                      : `${themeClasses.warning} ${themeClasses.warningBg}`
                  }`}>
                    {report.status}
                  </span>
                  
                  {report.status === 'Completed' && (
                    <button className={`${themeClasses.textMuted} hover:${themeClasses.textSecondary} p-0.5 transition-colors`}>
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  )}
                  
                  <button 
                    onClick={() => deleteReport(report.id)}
                    className={`${themeClasses.error} ${themeClasses.errorHover} p-0.5 transition-colors`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
          
          {reportHistory.length === 0 && (
            <div className={`px-4 py-8 text-center ${themeClasses.textMuted}`}>
              <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm font-medium mb-1">No reports yet</p>
              <p className="text-xs">Generate your first analysis report to see it here</p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default ReportHistorySection;