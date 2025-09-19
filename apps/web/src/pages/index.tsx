import { useState } from 'react'
import Head from 'next/head'
import { Search, Upload, Settings, BarChart3 } from 'lucide-react'

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery.trim()) return
    
    setIsSearching(true)
    try {
      // TODO: Implement search functionality
      console.log('Searching for:', searchQuery)
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <>
      <Head>
        <title>PartSync - Intelligent Part Replacement Engine</title>
        <meta name="description" content="Find perfect replacements for electronic components using AI-powered FFF scoring" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <h1 className="text-2xl font-bold text-gray-900">PartSync</h1>
                <span className="ml-2 px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full">
                  AI-Powered
                </span>
              </div>
              <nav className="flex space-x-8">
                <a href="#" className="text-gray-500 hover:text-gray-900">Parts</a>
                <a href="#" className="text-gray-500 hover:text-gray-900">BOM</a>
                <a href="#" className="text-gray-500 hover:text-gray-900">Ingest</a>
                <a href="#" className="text-gray-500 hover:text-gray-900">Settings</a>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Find Perfect Part Replacements
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Never lose weeks waiting on obsolete or out-of-stock parts. 
              Repair your BOM in seconds with our AI-powered FFF scoring system.
            </p>

            {/* Search Bar */}
            <form onSubmit={handleSearch} className="max-w-2xl mx-auto">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by MPN, manufacturer, or description..."
                  className="w-full px-6 py-4 text-lg border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  type="submit"
                  disabled={isSearching}
                  className="absolute right-2 top-2 p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                >
                  <Search className="w-6 h-6" />
                </button>
              </div>
            </form>
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <Search className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Smart Search</h3>
              <p className="text-gray-600">
                Find parts using natural language queries with semantic search powered by LlamaIndex.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">FFF Scoring</h3>
              <p className="text-gray-600">
                Get ranked replacement candidates with Form, Fit, and Function compatibility scores.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                <Upload className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">BOM Upload</h3>
              <p className="text-gray-600">
                Upload your entire BOM and get replacement suggestions for every component.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mb-4">
                <Settings className="w-6 h-6 text-orange-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Live Ingestion</h3>
              <p className="text-gray-600">
                Automatically fetch fresh datasheets from Mouser and extract specifications.
              </p>
            </div>
          </div>

          {/* Demo Section */}
          <div className="bg-white rounded-lg shadow-sm border p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">
              Try It Out
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-blue-600">1</span>
                </div>
                <h4 className="text-lg font-semibold mb-2">Search Parts</h4>
                <p className="text-gray-600">Try searching for "LM358" or "QFN LDO 3.3V"</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-green-600">2</span>
                </div>
                <h4 className="text-lg font-semibold mb-2">View Replacements</h4>
                <p className="text-gray-600">See ranked candidates with detailed FFF scores</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-purple-600">3</span>
                </div>
                <h4 className="text-lg font-semibold mb-2">Export Results</h4>
                <p className="text-gray-600">Download your updated BOM with replacements</p>
              </div>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center text-gray-500">
              <p>&copy; 2024 PartSync. Powered by LlamaIndex, LlamaParse, and Mouser API.</p>
            </div>
          </div>
        </footer>
      </div>
    </>
  )
}
