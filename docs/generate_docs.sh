#!/bin/bash

# XGrammar Doxygen Documentation Generator
# This script generates both public and private API documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

echo "🚀 XGrammar Documentation Generator"
echo "Project root: $PROJECT_ROOT"

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "❌ Error: Doxygen is not installed. Please install doxygen first."
    echo "   Ubuntu/Debian: sudo apt-get install doxygen graphviz"
    echo "   macOS: brew install doxygen graphviz"
    exit 1
fi

# Check if dot (graphviz) is available
if ! command -v dot &> /dev/null; then
    echo "⚠️  Warning: Graphviz (dot) is not installed. Class diagrams will be disabled."
    echo "   Ubuntu/Debian: sudo apt-get install graphviz"
    echo "   macOS: brew install graphviz"
    HAVE_DOT="NO"
else
    HAVE_DOT="YES"
    echo "✅ Graphviz found - enabling class diagrams"
fi

# Create documentation directories
mkdir -p docs/doxygen/public
mkdir -p docs/doxygen/private
mkdir -p docs/doxygen/combined
mkdir -p docs/doxygen-config

echo "📁 Created documentation directories"

# Generate Public API Documentation
echo "📚 Generating Public API Documentation..."
cat > docs/doxygen-config/Doxyfile.public << EOF
# Public API Documentation Configuration
@INCLUDE = docs/Doxyfile

PROJECT_NAME           = "XGrammar Public API"
PROJECT_BRIEF          = "Public API Documentation for XGrammar - Structured Generation Library"
OUTPUT_DIRECTORY       = docs/doxygen/public

# Public API specific settings
INPUT                  = include/xgrammar
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = NO
HIDE_UNDOC_MEMBERS     = YES
HIDE_UNDOC_CLASSES     = YES
INTERNAL_DOCS          = NO

# Include only public headers
FILE_PATTERNS          = *.h *.hpp

# Generate cleaner public docs
HAVE_DOT               = $HAVE_DOT
CLASS_GRAPH            = $HAVE_DOT
COLLABORATION_GRAPH    = $HAVE_DOT
INCLUDE_GRAPH          = $HAVE_DOT
INCLUDED_BY_GRAPH      = $HAVE_DOT

# Custom CSS for public docs
HTML_EXTRA_STYLESHEET  = docs/doxygen-config/public_style.css
EOF

# Generate Private/Internal API Documentation
echo "🔧 Generating Private API Documentation..."
cat > docs/doxygen-config/Doxyfile.private << EOF
# Private API Documentation Configuration
@INCLUDE = docs/Doxyfile

PROJECT_NAME           = "XGrammar Internal API"
PROJECT_BRIEF          = "Internal API Documentation for XGrammar - Implementation Details"
OUTPUT_DIRECTORY       = docs/doxygen/private

# Private API specific settings
INPUT                  = cpp
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
HIDE_UNDOC_MEMBERS     = NO
HIDE_UNDOC_CLASSES     = NO
INTERNAL_DOCS          = YES

# Include all source files
FILE_PATTERNS          = *.c *.cc *.cpp *.h *.hpp

# More detailed graphs for internal docs
HAVE_DOT               = $HAVE_DOT
CLASS_GRAPH            = $HAVE_DOT
COLLABORATION_GRAPH    = $HAVE_DOT
INCLUDE_GRAPH          = $HAVE_DOT
INCLUDED_BY_GRAPH      = $HAVE_DOT
CALL_GRAPH             = NO
CALLER_GRAPH           = NO

# Custom CSS for private docs
HTML_EXTRA_STYLESHEET  = docs/doxygen-config/private_style.css
EOF

# Generate Combined Documentation
echo "📖 Generating Combined Documentation..."
cat > docs/doxygen-config/Doxyfile.combined << EOF
# Combined Documentation Configuration
@INCLUDE = docs/Doxyfile

PROJECT_NAME           = "XGrammar Complete API"
PROJECT_BRIEF          = "Complete API Documentation for XGrammar - Public and Internal APIs"
OUTPUT_DIRECTORY       = docs/doxygen/combined

# Combined settings
INPUT                  = include/xgrammar cpp
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
HIDE_UNDOC_MEMBERS     = NO
HIDE_UNDOC_CLASSES     = NO
INTERNAL_DOCS          = YES

# Include all relevant files
FILE_PATTERNS          = *.c *.cc *.cpp *.h *.hpp

# Full feature graphs
HAVE_DOT               = $HAVE_DOT
CLASS_GRAPH            = $HAVE_DOT
COLLABORATION_GRAPH    = $HAVE_DOT
INCLUDE_GRAPH          = $HAVE_DOT
INCLUDED_BY_GRAPH      = $HAVE_DOT
GRAPHICAL_HIERARCHY    = $HAVE_DOT
DIRECTORY_GRAPH        = $HAVE_DOT

# Separate sections for public vs private
ENABLED_SECTIONS       = PUBLIC_API PRIVATE_API
EOF

# Create custom CSS files
echo "🎨 Creating custom stylesheets..."

# Public API stylesheet - clean and professional
cat > docs/doxygen-config/public_style.css << 'EOF'
/* Public API Custom Styles */
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.title {
    color: #2c3e50;
    font-weight: 600;
}

.groupheader {
    background-color: #3498db;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 10px 0 5px 0;
}

.memitem {
    border-left: 3px solid #3498db;
    margin-bottom: 10px;
}

.navpath {
    background-color: #ecf0f1;
}

/* Highlight public API elements */
.public {
    border-left: 4px solid #27ae60;
}
EOF

# Private API stylesheet - more technical feel
cat > docs/doxygen-config/private_style.css << 'EOF'
/* Private API Custom Styles */
.header {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
}

.title {
    color: #c0392b;
    font-weight: 600;
}

.groupheader {
    background-color: #e74c3c;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 10px 0 5px 0;
}

.memitem {
    border-left: 3px solid #e74c3c;
    margin-bottom: 10px;
}

.navpath {
    background-color: #fdf2f2;
}

/* Highlight private API elements */
.private {
    border-left: 4px solid #e67e22;
}

/* Warning styles for internal APIs */
.internal {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 8px;
    border-radius: 4px;
    margin: 5px 0;
}
EOF

# Function to run doxygen with progress
run_doxygen() {
    local config_file=$1
    local doc_type=$2

    echo "  Generating $doc_type documentation..."
    # Run from project root so paths work correctly
    cd "$PROJECT_ROOT/.."
    if doxygen "$config_file" > /dev/null 2>&1; then
        echo "  ✅ $doc_type documentation generated successfully"
        cd "$PROJECT_ROOT"
        return 0
    else
        echo "  ❌ Error generating $doc_type documentation"
        echo "  Check the configuration in $config_file"
        cd "$PROJECT_ROOT"
        return 1
    fi
}

# Generate documentation based on command line argument
case "${1:-all}" in
    "public")
        echo "Generating only Public API documentation..."
        run_doxygen "docs/doxygen-config/Doxyfile.public" "Public API"
        ;;
    "private")
        echo "Generating only Private API documentation..."
        run_doxygen "docs/doxygen-config/Doxyfile.private" "Private API"
        ;;
    "combined")
        echo "Generating only Combined documentation..."
        run_doxygen "docs/doxygen-config/Doxyfile.combined" "Combined"
        ;;
    "all"|*)
        echo "Generating all documentation types..."
        run_doxygen "docs/doxygen-config/Doxyfile.public" "Public API"
        run_doxygen "docs/doxygen-config/Doxyfile.private" "Private API"
        run_doxygen "docs/doxygen-config/Doxyfile.combined" "Combined"
        ;;
esac

# Create index page
echo "📄 Creating documentation index..."
cat > docs/doxygen/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>XGrammar Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .docs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 30px; }
        .doc-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; text-decoration: none; transition: transform 0.2s; }
        .doc-card:hover { transform: translateY(-2px); }
        .doc-card h3 { margin: 0 0 10px 0; }
        .doc-card p { margin: 0; opacity: 0.9; }
        .footer { margin-top: 40px; text-align: center; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>XGrammar Documentation</h1>
        <p>Welcome to the XGrammar documentation. Choose the documentation type that best fits your needs:</p>

        <div class="docs-grid">
            <a href="public/html/index.html" class="doc-card">
                <h3>📚 Public API</h3>
                <p>User-facing API documentation for application developers</p>
            </a>

            <a href="private/html/index.html" class="doc-card">
                <h3>🔧 Internal API</h3>
                <p>Implementation details for contributors and maintainers</p>
            </a>

            <a href="combined/html/index.html" class="doc-card">
                <h3>📖 Complete Reference</h3>
                <p>Comprehensive documentation including all APIs</p>
            </a>
        </div>

        <div class="footer">
            <p>Generated with Doxygen • XGrammar v0.1.19</p>
        </div>
    </div>
</body>
</html>
EOF

echo ""
echo "🎉 Documentation generation complete!"
echo ""
echo "📍 Documentation locations:"
echo "   • Main index: docs/doxygen/index.html"
echo "   • Public API: docs/doxygen/public/html/index.html"
echo "   • Private API: docs/doxygen/private/html/index.html"
echo "   • Combined: docs/doxygen/combined/html/index.html"
echo ""
echo "🌐 To view locally, open docs/doxygen/index.html in your browser"
echo "   or run: python3 -m http.server 8000 (then visit http://localhost:8000/docs/doxygen/)"
echo ""
echo "💡 Usage:"
echo "   ./generate_docs.sh        # Generate all documentation"
echo "   ./generate_docs.sh public # Generate only public API docs"
echo "   ./generate_docs.sh private# Generate only private API docs"
echo "   ./generate_docs.sh combined# Generate only combined docs"
