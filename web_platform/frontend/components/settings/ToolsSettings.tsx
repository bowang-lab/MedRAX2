/**
 * ToolsSettings Component
 * 
 * Manage medical imaging tools:
 * - View available tools grouped by category
 * - Load/unload tools dynamically
 * - View tool status, dependencies, and info
 * - Install all dependencies button
 */

'use client';

import { useState, useEffect, useRef } from 'react';
import { Wrench, Loader2, Info, Download, Check, X, AlertCircle } from 'lucide-react';
import { getTools, loadTool, unloadTool } from '../../lib/api/toolManagement';
import { useToolLoadingSSE } from '../../lib/hooks/useToolLoadingSSE';
import { ToolLoadingProgress } from '../tools/ToolLoadingProgress';
import { API_CONFIG } from '../../lib/config/api';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';
import { Badge } from '../ui/Badge';

interface Tool {
    id: string;
    name: string;
    description: string;
    status: 'available' | 'unavailable' | 'loaded' | 'unloaded' | 'error' | 'loading';
    category: string;
    loaded_at?: string;
    dependencies?: string[];
    requires_gpu?: boolean;
    error_message?: string;
}

interface ToolsByCategory {
    [category: string]: Tool[];
}

const CATEGORY_DISPLAY_NAMES: { [key: string]: string } = {
    'classification': 'Classification',
    'vqa': 'Visual Question Answering',
    'segmentation': 'Segmentation',
    'generation': 'Generation',
    'grounding': 'Grounding',
    'processing': 'Image Processing',
    'retrieval': 'Retrieval & Search',
    'execution': 'Code Execution',
};

const CATEGORY_DESCRIPTIONS: { [key: string]: string } = {
    'classification': 'Identify pathologies and conditions in medical images',
    'vqa': 'Answer questions about medical images using AI',
    'segmentation': 'Segment and identify regions in medical images',
    'generation': 'Generate reports and synthetic medical images',
    'grounding': 'Locate specific findings in medical images',
    'processing': 'Process and convert medical image formats',
    'retrieval': 'Search medical knowledge and web resources',
    'execution': 'Execute Python code for custom analysis',
};

export function ToolsSettings() {
    const [tools, setTools] = useState<Tool[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
    const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
    const [loadingToolId, setLoadingToolId] = useState<string | null>(null);
    const [loadingProgress, setLoadingProgress] = useState<{ progress: number; message: string } | null>(null);
    
    // Store EventSource ref for cleanup on unmount
    const eventSourceRef = useRef<EventSource | null>(null);

    useEffect(() => {
        loadTools();
        // Expand all categories by default
        setExpandedCategories(new Set(Object.keys(CATEGORY_DISPLAY_NAMES)));
    }, []);

    // Cleanup EventSource on unmount to prevent memory leak
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, []);

    // NO MORE POLLING! SSE handles real-time updates now

    const loadTools = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const fetchedTools = await getTools();
            setTools(fetchedTools);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load tools');
        } finally {
            setIsLoading(false);
        }
    };

    const handleLoadTool = async (toolId: string) => {
        // Close any existing connection first
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        
        // Update UI immediately
        setTools(tools.map(t =>
            t.id === toolId ? { ...t, status: 'loading' as const } : t
        ));
        
        setLoadingToolId(toolId);
        setLoadingProgress({ progress: 0, message: 'Connecting...' });

        // Use SSE for real-time progress updates
        const token = localStorage.getItem('token');
        if (!token) {
            setError('Authentication required');
            setLoadingToolId(null);
            setLoadingProgress(null);
            return;
        }

        // Use centralized API config
        const eventSource = new EventSource(
            `${API_CONFIG.baseURL}/api/tools/${toolId}/load-stream?token=${encodeURIComponent(token)}`
        );
        
        // Store in ref for cleanup
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.status === 'loading') {
                    setLoadingProgress({ progress: data.progress, message: data.message });
                } else if (data.status === 'loaded') {
                    setLoadingProgress({ progress: 100, message: 'Complete!' });
                    eventSource.close();
                    eventSourceRef.current = null;
                    setLoadingToolId(null);
                    setLoadingProgress(null);
                    // Refresh tools list once
                    loadTools();
                } else if (data.status === 'error') {
                    setError(data.message);
                    eventSource.close();
                    eventSourceRef.current = null;
                    setLoadingToolId(null);
                    setLoadingProgress(null);
                    loadTools();
                }
            } catch (err) {
                console.error('Failed to parse SSE message:', err);
            }
        };

        eventSource.onerror = (err) => {
            console.error('SSE connection error:', err);
            setError('Connection error. Please try again.');
            eventSource.close();
            eventSourceRef.current = null;
            setLoadingToolId(null);
            setLoadingProgress(null);
            loadTools();
        };
    };

    const handleUnloadTool = async (toolId: string) => {
        // Update UI immediately
        setTools(tools.map(t =>
            t.id === toolId ? { ...t, status: 'loading' as const } : t
        ));

        try {
            await unloadTool(toolId);
            await loadTools(); // Refresh all tools
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to unload tool');
            await loadTools();
        }
    };

    const toggleCategory = (category: string) => {
        const newExpanded = new Set(expandedCategories);
        if (newExpanded.has(category)) {
            newExpanded.delete(category);
        } else {
            newExpanded.add(category);
        }
        setExpandedCategories(newExpanded);
    };

    const groupToolsByCategory = (): ToolsByCategory => {
        return tools.reduce((acc, tool) => {
            const category = tool.category || 'other';
            if (!acc[category]) {
                acc[category] = [];
            }
            acc[category].push(tool);
            return acc;
        }, {} as ToolsByCategory);
    };

    const getStatusBadgeVariant = (status: string): 'success' | 'error' | 'warning' | 'info' | 'default' => {
        switch (status) {
            case 'loaded': return 'success';
            case 'error': return 'error';
            case 'unavailable': return 'warning';
            case 'loading': return 'info';
            default: return 'default';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'loaded': return <Check className="h-4 w-4" />;
            case 'error': return <X className="h-4 w-4" />;
            case 'unavailable': return <AlertCircle className="h-4 w-4" />;
            case 'loading': return <Loader2 className="h-4 w-4 animate-spin" />;
            default: return null;
        }
    };

    const getToolStats = () => {
        const available = tools.filter(t => t.status === 'available').length;
        const loaded = tools.filter(t => t.status === 'loaded').length;
        const unavailable = tools.filter(t => t.status === 'unavailable').length;
        return { total: tools.length, available, loaded, unavailable };
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-12">
                <Spinner size="lg" />
            </div>
        );
    }

    const toolsByCategory = groupToolsByCategory();
    const stats = getToolStats();

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Wrench className="h-6 w-6" />
                        Tools Management
                    </h2>
                    <p className="mt-1 text-sm text-zinc-400">
                        Load and unload medical imaging analysis tools dynamically
                    </p>
                </div>
                <Button
                    variant="secondary"
                    size="sm"
                    onClick={loadTools}
                >
                    Refresh
                </Button>
            </div>

            {/* Stats */}
            <Card className="p-4 bg-zinc-800/50">
                <div className="grid grid-cols-4 gap-4">
                    <div>
                        <div className="text-2xl font-bold text-white">{stats.total}</div>
                        <div className="text-sm text-zinc-400">Total Tools</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-emerald-500">{stats.available}</div>
                        <div className="text-sm text-zinc-400">Available</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-blue-500">{stats.loaded}</div>
                        <div className="text-sm text-zinc-400">Loaded</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-amber-500">{stats.unavailable}</div>
                        <div className="text-sm text-zinc-400">Unavailable</div>
                    </div>
                </div>
            </Card>

            {/* Installation Info */}
            {stats.unavailable > 0 && (
                <Card className="p-4 bg-amber-900/20 border border-amber-800">
                    <div className="flex items-start gap-3">
                        <Download className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                            <h3 className="font-semibold text-amber-200">Missing Dependencies</h3>
                            <p className="text-sm text-amber-300/80 mt-1">
                                {stats.unavailable} tool{stats.unavailable !== 1 ? 's are' : ' is'} unavailable due to missing dependencies.
                            </p>
                            <div className="mt-2 text-sm text-amber-300/80 font-mono bg-black/30 p-2 rounded">
                                cd backend && source venv/bin/activate && pip install -r requirements.txt
                            </div>
                            <p className="text-xs text-amber-400/70 mt-2">
                                After installation, restart the backend server for changes to take effect.
                            </p>
                        </div>
                    </div>
                </Card>
            )}

            {/* Error Display */}
            {error && (
                <Card className="p-4 bg-red-900/20 border border-red-800">
                    <p className="text-red-400">{error}</p>
                </Card>
            )}

            {/* Tools by Category */}
            {(!tools || tools.length === 0) ? (
                <Card className="p-8 text-center">
                    <p className="text-zinc-500">No tools available</p>
                </Card>
            ) : (
                <div className="space-y-4">
                    {Object.entries(toolsByCategory).map(([category, categoryTools]) => (
                        <Card key={category} className="overflow-hidden">
                            {/* Category Header */}
                            <button
                                onClick={() => toggleCategory(category)}
                                className="w-full px-6 py-4 flex items-center justify-between bg-zinc-800/50 hover:bg-zinc-800/70 transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="text-left">
                                        <h3 className="font-semibold text-white">
                                            {CATEGORY_DISPLAY_NAMES[category] || category}
                                        </h3>
                                        <p className="text-sm text-zinc-400">
                                            {CATEGORY_DESCRIPTIONS[category] || ''} ({categoryTools.length} tools)
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Badge variant="default" size="sm">
                                        {categoryTools.filter(t => t.status === 'loaded').length} loaded
                                    </Badge>
                                    <div className={`transform transition-transform ${expandedCategories.has(category) ? 'rotate-180' : ''}`}>
                                        ▼
                                    </div>
                                </div>
                            </button>

                            {/* Category Tools */}
                            {expandedCategories.has(category) && (
                                <div className="p-6 space-y-4">
                                    {categoryTools.map((tool) => (
                                        <div
                                            key={tool.id}
                                            className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors"
                                        >
                                            <div className="flex items-start justify-between gap-4">
                                                {/* Tool Info */}
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <h4 className="font-semibold text-white">
                                                            {tool.name}
                                                        </h4>
                                                        <Badge variant={getStatusBadgeVariant(tool.status)} size="sm">
                                                            <span className="flex items-center gap-1">
                                                                {getStatusIcon(tool.status)}
                                                                {tool.status}
                                                            </span>
                                                        </Badge>
                                                        {tool.requires_gpu && (
                                                            <Badge variant="default" size="sm">GPU Required</Badge>
                                                        )}
                                                    </div>

                                                    <p className="text-sm text-zinc-400 mb-2">
                                                        {tool.description}
                                                    </p>

                                                    {/* Real-time SSE progress bar */}
                                                    {loadingToolId === tool.id && loadingProgress && (
                                                        <div className="mb-3">
                                                            <ToolLoadingProgress
                                                                progress={loadingProgress.progress}
                                                                message={loadingProgress.message}
                                                            />
                                                        </div>
                                                    )}

                                                    {tool.loaded_at && (
                                                        <p className="text-xs text-zinc-500">
                                                            Loaded at: {new Date(tool.loaded_at).toLocaleString()}
                                                        </p>
                                                    )}

                                                    {tool.error_message && (
                                                        <p className="text-xs text-amber-400 mt-1">
                                                            {tool.error_message}
                                                        </p>
                                                    )}

                                                    {tool.dependencies && tool.dependencies.length > 0 && (
                                                        <div className="mt-2">
                                                            <button
                                                                onClick={() => setSelectedTool(selectedTool?.id === tool.id ? null : tool)}
                                                                className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                                            >
                                                                <Info className="h-3 w-3" />
                                                                Show dependencies ({tool.dependencies.length})
                                                            </button>
                                                            {selectedTool?.id === tool.id && (
                                                                <div className="mt-2 p-2 bg-black/30 rounded text-xs text-zinc-400 font-mono">
                                                                    {tool.dependencies.join(', ')}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Action Buttons */}
                                                <div className="flex items-center gap-2">
                                                    {tool.status === 'loading' ? (
                                                        <Button variant="secondary" size="sm" disabled>
                                                            <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                                            Loading...
                                                        </Button>
                                                    ) : tool.status === 'loaded' ? (
                                                        <Button
                                                            variant="secondary"
                                                            size="sm"
                                                            onClick={() => handleUnloadTool(tool.id)}
                                                        >
                                                            Unload
                                                        </Button>
                                                    ) : tool.status === 'unavailable' ? (
                                                        <Button
                                                            variant="secondary"
                                                            size="sm"
                                                            disabled
                                                            title="Install dependencies first"
                                                        >
                                                            Unavailable
                                                        </Button>
                                                    ) : (
                                                        <Button
                                                            variant="primary"
                                                            size="sm"
                                                            onClick={() => handleLoadTool(tool.id)}
                                                            disabled={loadingToolId !== null}
                                                            title={loadingToolId !== null ? 'Another tool is loading' : undefined}
                                                        >
                                                            Load
                                                        </Button>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
}
