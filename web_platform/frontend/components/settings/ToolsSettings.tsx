/**
 * ToolsSettings Component
 * 
 * Manage medical imaging tools:
 * - View available tools grouped by category
 * - Load/unload tools dynamically with real-time SSE progress
 * - View tool status, dependencies, and info
 * - Bulk load tools with SSE for each tool
 */

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Wrench, Loader2, Info, Download, Check, X, AlertCircle } from 'lucide-react';
import { getTools, loadTool, unloadTool, bulkLoadTools, Tool } from '../../lib/api/toolManagement';
import { ToolLoadingProgress } from '../tools/ToolLoadingProgress';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';
import { Badge } from '../ui/Badge';
import { API_CONFIG } from '../../lib/config/api';
import { AUTH_CONFIG } from '../../lib/config/app';

interface ToolsByCategory {
    [category: string]: Tool[];
}

interface ToolLoadingState {
    progress: number;
    message: string;
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
    const [selectedToolIds, setSelectedToolIds] = useState<Set<string>>(new Set());

    // Track loading tools with SSE connections
    const [loadingTools, setLoadingTools] = useState<Map<string, ToolLoadingState>>(new Map());
    const sseConnectionsRef = useRef<Map<string, EventSource>>(new Map());

    useEffect(() => {
        loadTools();
        // Expand all categories by default
        setExpandedCategories(new Set(Object.keys(CATEGORY_DISPLAY_NAMES)));

        // Capture the current ref value for cleanup
        const sseConnections = sseConnectionsRef.current;

        // Cleanup SSE connections on unmount
        return () => {
            sseConnections.forEach(connection => connection.close());
            sseConnections.clear();
        };
    }, []);

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

    // Create SSE connection for a specific tool
    const createSSEConnection = useCallback((toolId: string) => {
        // Don't create duplicate connections
        if (sseConnectionsRef.current.has(toolId)) {
            return;
        }

        const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
        if (!token) {
            console.error('No auth token available');
            return;
        }

        const url = `${API_CONFIG.baseURL}/api/tools/${toolId}/load-stream?token=${encodeURIComponent(token)}`;
        const eventSource = new EventSource(url);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.status === 'loading') {
                    setLoadingTools(prev => new Map(prev).set(toolId, {
                        progress: data.progress,
                        message: data.message
                    }));
                } else if (data.status === 'loaded') {
                    // Tool loaded successfully
                    console.log(`✅ Tool ${toolId} loaded successfully`);
                    setLoadingTools(prev => {
                        const next = new Map(prev);
                        next.delete(toolId);
                        return next;
                    });
                    eventSource.close();
                    sseConnectionsRef.current.delete(toolId);
                    loadTools(); // Refresh to get final state
                } else if (data.status === 'error') {
                    console.error(`❌ Tool ${toolId} loading error:`, data.message);
                    setError(`${toolId}: ${data.message}`);
                    setLoadingTools(prev => {
                        const next = new Map(prev);
                        next.delete(toolId);
                        return next;
                    });
                    eventSource.close();
                    sseConnectionsRef.current.delete(toolId);
                    loadTools(); // Refresh to get error state
                }
            } catch (err) {
                console.error('Failed to parse SSE message:', err);
            }
        };

        eventSource.onerror = (err) => {
            console.error(`SSE connection error for ${toolId}:`, err);
            setLoadingTools(prev => {
                const next = new Map(prev);
                next.delete(toolId);
                return next;
            });
            eventSource.close();
            sseConnectionsRef.current.delete(toolId);
            loadTools();
        };

        sseConnectionsRef.current.set(toolId, eventSource);
    }, []);

    const handleLoadTool = async (toolId: string) => {
        try {
            setError(null);

            // Update UI immediately
            setTools(tools.map(t =>
                t.id === toolId ? { ...t, status: 'loading' as const } : t
            ));

            // Start SSE connection BEFORE calling load
            setLoadingTools(prev => new Map(prev).set(toolId, {
                progress: 0,
                message: 'Initiating load...'
            }));

            // Initiate the actual load request (backend starts background loading)
            await loadTool(toolId);

            // Create SSE connection to track progress
            createSSEConnection(toolId);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load tool');
            setLoadingTools(prev => {
                const next = new Map(prev);
                next.delete(toolId);
                return next;
            });
            loadTools();
        }
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

    const handleBulkLoadAll = async () => {
        try {
            setError(null);

            // Call bulk load endpoint
            const result = await bulkLoadTools({ loadAll: true });

            // For each tool that started loading, create an SSE connection
            const loadingToolIds = result.results
                .filter(r => r.success && r.status === 'loading')
                .map(r => r.id);

            console.log(`📡 Starting SSE for ${loadingToolIds.length} tools:`, loadingToolIds);

            // Update UI to show loading state
            setTools(prev => prev.map(t =>
                loadingToolIds.includes(t.id) ? { ...t, status: 'loading' as const } : t
            ));

            // Create SSE connection for each loading tool
            loadingToolIds.forEach(toolId => {
                setLoadingTools(prev => new Map(prev).set(toolId, {
                    progress: 0,
                    message: 'Starting...'
                }));
                createSSEConnection(toolId);
            });

            // Immediate refresh to get accurate states
            await loadTools();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to bulk load tools');
        }
    };

    const handleBulkLoadSelected = async (ids: string[]) => {
        try {
            setError(null);

            // Call bulk load endpoint
            const result = await bulkLoadTools({ toolIds: ids });

            // For each tool that started loading, create an SSE connection
            const loadingToolIds = result.results
                .filter(r => r.success && r.status === 'loading')
                .map(r => r.id);

            console.log(`📡 Starting SSE for ${loadingToolIds.length} tools:`, loadingToolIds);

            // Update UI to show loading state
            setTools(prev => prev.map(t =>
                loadingToolIds.includes(t.id) ? { ...t, status: 'loading' as const } : t
            ));

            // Create SSE connection for each loading tool
            loadingToolIds.forEach(toolId => {
                setLoadingTools(prev => new Map(prev).set(toolId, {
                    progress: 0,
                    message: 'Starting...'
                }));
                createSSEConnection(toolId);
            });

            // Immediate refresh to get accurate states
            await loadTools();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to bulk load selected tools');
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

    const toggleSelectTool = (toolId: string) => {
        setSelectedToolIds(prev => {
            const next = new Set(prev);
            if (next.has(toolId)) next.delete(toolId); else next.add(toolId);
            return next;
        });
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
        const loading = tools.filter(t => t.status === 'loading').length;
        return { total: tools.length, available, loaded, unavailable, loading };
    };

    if (isLoading && tools.length === 0) {
        return (
            <div className="flex items-center justify-center py-12">
                <Spinner size="lg" />
            </div>
        );
    }

    const toolsByCategory = groupToolsByCategory();
    const stats = getToolStats();
    const hasActiveSSE = loadingTools.size > 0;

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
                <div className="flex items-center gap-2">
                    <Button
                        variant="secondary"
                        size="sm"
                        onClick={loadTools}
                        disabled={hasActiveSSE}
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="primary"
                        size="sm"
                        onClick={handleBulkLoadAll}
                        disabled={hasActiveSSE}
                        title={hasActiveSSE ? 'Tools are currently loading' : 'Load all available tools'}
                    >
                        Load All
                    </Button>
                    <Button
                        variant="primary"
                        size="sm"
                        onClick={() => handleBulkLoadSelected(Array.from(selectedToolIds))}
                        disabled={selectedToolIds.size === 0 || hasActiveSSE}
                        title={selectedToolIds.size === 0 ? 'Select tools to load' : hasActiveSSE ? 'Tools are currently loading' : 'Load selected tools'}
                    >
                        Load Selected ({selectedToolIds.size})
                    </Button>
                </div>
            </div>

            {/* Stats */}
            <Card className="p-4 bg-zinc-800/50">
                <div className="grid grid-cols-5 gap-4">
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
                        <div className="text-2xl font-bold text-purple-500">{stats.loading}</div>
                        <div className="text-sm text-zinc-400">Loading</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-amber-500">{stats.unavailable}</div>
                        <div className="text-sm text-zinc-400">Unavailable</div>
                    </div>
                </div>
            </Card>

            {/* Real-time SSE Loading Indicator */}
            {hasActiveSSE && (
                <Card className="p-4 bg-blue-900/20 border border-blue-800">
                    <div className="flex items-center gap-3">
                        <Loader2 className="h-5 w-5 text-blue-500 animate-spin flex-shrink-0" />
                        <div className="flex-1">
                            <h3 className="font-semibold text-blue-200">Real-time Tool Loading (SSE)</h3>
                            <p className="text-sm text-blue-300/80 mt-1">
                                {loadingTools.size} tool{loadingTools.size !== 1 ? 's are' : ' is'} loading with live progress updates via Server-Sent Events.
                            </p>
                            <p className="text-xs text-blue-400/70 mt-2">
                                ⓘ Progress shows estimated completion. First-time downloads may take several minutes.
                            </p>
                        </div>
                    </div>
                </Card>
            )}

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
                    <div className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                        <p className="text-red-400">{error}</p>
                    </div>
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
                                    {categoryTools.map((tool) => {
                                        const toolLoadingState = loadingTools.get(tool.id);
                                        const hasSSE = toolLoadingState !== undefined;

                                        return (
                                            <div
                                                key={tool.id}
                                                className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors"
                                            >
                                                <div className="flex items-start justify-between gap-4">
                                                    {/* Tool Info */}
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <input
                                                                type="checkbox"
                                                                className="h-4 w-4 accent-blue-500"
                                                                checked={selectedToolIds.has(tool.id)}
                                                                onChange={() => toggleSelectTool(tool.id)}
                                                                disabled={tool.status === 'loading' || tool.status === 'loaded'}
                                                                aria-label={`Select ${tool.name}`}
                                                            />
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
                                                            {hasSSE && (
                                                                <Badge variant="info" size="sm">📡 SSE</Badge>
                                                            )}
                                                        </div>

                                                        <p className="text-sm text-zinc-400 mb-2">
                                                            {tool.description}
                                                        </p>

                                                        {/* Real-time SSE progress bar */}
                                                        {hasSSE && (
                                                            <div className="mb-3">
                                                                <ToolLoadingProgress
                                                                    progress={toolLoadingState.progress}
                                                                    message={toolLoadingState.message}
                                                                />
                                                                <p className="text-xs text-zinc-500 mt-1">
                                                                    ⓘ Real-time progress via SSE (estimated completion)
                                                                </p>
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
                                                                disabled={hasActiveSSE}
                                                                title={hasActiveSSE ? 'Another tool is loading' : undefined}
                                                            >
                                                                Load
                                                            </Button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
}
