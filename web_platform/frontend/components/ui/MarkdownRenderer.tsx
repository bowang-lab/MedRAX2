'use client';

/**
 * Markdown Renderer Component
 * 
 * Renders markdown content with syntax highlighting.
 */

import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import 'highlight.js/styles/github-dark.css';

export interface MarkdownRendererProps {
    /** Markdown content to render */
    content: string;
    /** Additional CSS classes */
    className?: string;
}

/**
 * Renders markdown content with GitHub Flavored Markdown and syntax highlighting.
 */
export function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
    return (
        <div className={`markdown-content ${className}`}>
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                    // Custom heading styles
                    h1: (props) => (
                        <h1 className="text-2xl font-bold mt-6 mb-4 text-white" {...props} />
                    ),
                    h2: (props) => (
                        <h2 className="text-xl font-bold mt-5 mb-3 text-white" {...props} />
                    ),
                    h3: (props) => (
                        <h3 className="text-lg font-semibold mt-4 mb-2 text-white" {...props} />
                    ),

                    // Paragraph styles
                    p: (props) => (
                        <p className="mb-4 text-zinc-300 leading-relaxed" {...props} />
                    ),

                    // List styles
                    ul: (props) => (
                        <ul className="list-disc list-inside mb-4 space-y-2 text-zinc-300" {...props} />
                    ),
                    ol: (props) => (
                        <ol className="list-decimal list-inside mb-4 space-y-2 text-zinc-300" {...props} />
                    ),
                    li: (props) => (
                        <li className="ml-4" {...props} />
                    ),

                    // Code blocks
                    code: ({ inline, className, children, ...props }: { inline?: boolean; className?: string; children?: React.ReactNode }) => {
                        return inline ? (
                            <code
                                className="bg-zinc-800 text-emerald-400 px-1.5 py-0.5 rounded text-sm font-mono"
                                {...props}
                            >
                                {children}
                            </code>
                        ) : (
                            <code
                                className={`${className || ''} block bg-zinc-900 p-4 rounded-lg overflow-x-auto text-sm font-mono`}
                                {...props}
                            >
                                {children}
                            </code>
                        );
                    },

                    // Blockquote
                    blockquote: (props) => (
                        <blockquote
                            className="border-l-4 border-emerald-500 pl-4 italic text-zinc-400 my-4"
                            {...props}
                        />
                    ),

                    // Links
                    a: (props) => (
                        <a
                            className="text-emerald-400 hover:text-emerald-300 underline"
                            target="_blank"
                            rel="noopener noreferrer"
                            {...props}
                        />
                    ),

                    // Tables
                    table: (props) => (
                        <div className="overflow-x-auto my-4">
                            <table className="min-w-full border-collapse border border-zinc-700" {...props} />
                        </div>
                    ),
                    thead: (props) => (
                        <thead className="bg-zinc-800" {...props} />
                    ),
                    th: (props) => (
                        <th className="border border-zinc-700 px-4 py-2 text-left font-semibold text-white" {...props} />
                    ),
                    td: (props) => (
                        <td className="border border-zinc-700 px-4 py-2 text-zinc-300" {...props} />
                    ),

                    // Horizontal rule
                    hr: (props) => (
                        <hr className="my-6 border-t border-zinc-700" {...props} />
                    ),
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
}

