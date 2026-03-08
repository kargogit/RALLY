#!/usr/bin/env python3
"""
Graph Extraction Script - Production Grade

This script parses binary analysis artifacts to produce a unified JSON representation
of the program's Call Graph and Control Flow Graphs with human-readable labels.

File Hierarchy Expected:
    .
    ├── functions/
    │   └── 0x<addr>/
    │       ├── structuse.dot    (CFG for function at <addr>)
    │       └── structuse.txt
    ├── nasm/
    │   └── structuse.asm        (Assembly with symbol information)
    ├── structuse.callgraph.dot  (Call Graph)
    └── output.json              (Generated output)

Usage:
    python graph_extractor.py [base_directory] [output_file]
    
    Defaults: base_directory='.', output_file='graph_output.json'
"""

import json
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class BasicBlock:
    """Represents a basic block in the CFG"""
    address: str
    label: str
    end_address: Optional[str] = None

@dataclass
class Function:
    """Represents a function with its basic blocks"""
    name: str
    address: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Assembly Parser
# -----------------------------------------------------------------------------

class AsmParser:
    """
    Parses the Assembly file to extract function names, basic block labels,
    and their corresponding memory addresses.
    """
    
    # Match: ; Entry 10f0; block 1; address 10fd
    ENTRY_COMMENT_REGEX = re.compile(
        r';\s*Entry\s+([0-9a-fA-F]+);\s*block\s+(\d+);\s*address\s+([0-9a-fA-F]+)'
    )
    
    # Match: _init: or L1000_1:
    LABEL_REGEX = re.compile(r'^([_\.a-zA-Z0-9]+):\s*$')

    def __init__(self, asm_path: Path):
        self.asm_path = asm_path
        self.functions: Dict[str, Function] = {}  # Key: normalized address
        self.address_to_label: Dict[str, str] = {}  # Key: normalized address
        self.address_to_func_name: Dict[str, str] = {}  # Key: normalized address

    def _normalize_addr(self, addr: str) -> str:
        """Normalize address to lowercase hex without 0x prefix"""
        return addr.lower().replace('0x', '')

    def parse(self) -> bool:
        logger.info(f"Parsing Assembly file: {self.asm_path}")
        
        if not self.asm_path.exists():
            logger.error(f"Assembly file not found: {self.asm_path}")
            return False

        try:
            with open(self.asm_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read assembly file: {e}")
            return False

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for Entry Comment
            match_comment = self.ENTRY_COMMENT_REGEX.search(line)
            if match_comment:
                func_addr = self._normalize_addr(match_comment.group(1))
                block_addr = self._normalize_addr(match_comment.group(3))

                # Look ahead for the label
                label = None
                j = i + 1
                while j < min(i + 5, len(lines)):  # Look ahead max 5 lines
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith(';'):
                        j += 1
                        continue
                    
                    match_label = self.LABEL_REGEX.match(next_line)
                    if match_label:
                        label = match_label.group(1)
                        break
                    j += 1
                
                if label:
                    self.address_to_label[block_addr] = label
                    
                    # Track function starts (block 0 typically)
                    if func_addr not in self.address_to_func_name:
                        self.address_to_func_name[func_addr] = label
                        self.functions[func_addr] = Function(
                            name=label,
                            address=func_addr
                        )
                    
                    if func_addr in self.functions:
                        self.functions[func_addr].blocks[block_addr] = BasicBlock(
                            address=block_addr,
                            label=label
                        )
            
            i += 1

        logger.info(f"Extracted {len(self.functions)} functions from assembly")
        return True

# -----------------------------------------------------------------------------
# Call Graph DOT Parser
# -----------------------------------------------------------------------------

class CallGraphDotParser:
    """
    Parses the Call Graph DOT file to extract relationships between functions.
    """
    
    # Match node with HTML label: structuse_1000 [shape=plaintext,label=<<TABLE...>_init</FONT>...>>]
    NODE_REGEX = re.compile(
        r'(\w+)\s*\[.*?label=<<TABLE[^>]*>.*?<FONT[^>]*>([^<]+)</FONT>.*?>>\s*\]',
        re.DOTALL
    )
    
    # Match edge: structuse_10f0 -> structuse_1080
    EDGE_REGEX = re.compile(r'(\w+)\s*->\s*(\w+)')

    def __init__(self, dot_path: Path):
        self.dot_path = dot_path
        self.nodes: Dict[str, str] = {}  # Key: NodeID, Value: FunctionName
        self.node_addrs: Dict[str, str] = {}  # Key: NodeID, Value: Address
        self.edges: List[Tuple[str, str]] = []

    def _extract_addr_from_id(self, node_id: str) -> Optional[str]:
        """Extracts hex address from node ID like structuse_1000"""
        match = re.search(r'([0-9a-fA-F]+)$', node_id)
        return match.group(1).lower() if match else None

    def parse(self) -> bool:
        logger.info(f"Parsing Call Graph DOT: {self.dot_path}")
        
        if not self.dot_path.exists():
            logger.error(f"Call Graph DOT not found: {self.dot_path}")
            return False

        try:
            content = self.dot_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read DOT file: {e}")
            return False

        # Find Nodes
        for match in self.NODE_REGEX.finditer(content):
            node_id = match.group(1)
            label = match.group(2).strip()
            addr = self._extract_addr_from_id(node_id)
            
            self.nodes[node_id] = label
            if addr:
                self.node_addrs[node_id] = addr

        # Find Edges
        for match in self.EDGE_REGEX.finditer(content):
            self.edges.append((match.group(1), match.group(2)))

        logger.info(f"Extracted {len(self.nodes)} nodes and {len(self.edges)} edges")
        return True

# -----------------------------------------------------------------------------
# CFG DOT Parser
# -----------------------------------------------------------------------------

class CfgDotParser:
    """
    Parses individual Control Flow Graph DOT files for specific functions.
    """
    
    # Match node: structuse_0 [ ... label="0 [10f0,10fb]" ]
    CFG_NODE_REGEX = re.compile(
        r'(\w+)\s*\[.*?label="(\d+)\s*\[([0-9a-fA-F]+),([0-9a-fA-F]+)\]".*?\]'
    )
    
    # Match edge
    CFG_EDGE_REGEX = re.compile(r'(\w+)\s*->\s*(\w+)')

    def __init__(self, dot_path: Path, func_addr: str):
        self.dot_path = dot_path
        self.func_addr = func_addr.lower().replace('0x', '')
        self.nodes: Dict[str, Dict[str, str]] = {}  # Key: NodeID
        self.edges: List[Tuple[str, str]] = []

    def parse(self) -> bool:
        if not self.dot_path.exists():
            logger.debug(f"CFG DOT not found: {self.dot_path}")
            return False
        
        try:
            content = self.dot_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.debug(f"Failed to read CFG DOT: {e}")
            return False

        for match in self.CFG_NODE_REGEX.finditer(content):
            node_id = match.group(1)
            start_addr = match.group(3).lower()
            end_addr = match.group(4).lower()
            self.nodes[node_id] = {'start': start_addr, 'end': end_addr}

        for match in self.CFG_EDGE_REGEX.finditer(content):
            self.edges.append((match.group(1), match.group(2)))
            
        logger.debug(f"CFG {self.func_addr}: {len(self.nodes)} nodes, {len(self.edges)} edges")
        return True

# -----------------------------------------------------------------------------
# Graph Builder
# -----------------------------------------------------------------------------

class GraphBuilder:
    """
    Correlates data from all parsers to build the final JSON structure.
    """
    
    def __init__(
        self,
        asm_parser: AsmParser,
        cg_parser: CallGraphDotParser,
        functions_dir: Path
    ):
        self.asm = asm_parser
        self.cg = cg_parser
        self.functions_dir = functions_dir
        self.call_graph_adj: Dict[str, List[str]] = {}
        self.cfg_adj: Dict[str, Dict[str, List[str]]] = {}

    def build_call_graph(self):
        """Builds Call Graph using ASM function names, mapping DOT edges to names"""
        logger.info("Building Call Graph...")
        
        # Map NodeID -> Function Name (prefer ASM name, fallback to DOT label)
        id_to_name = {}
        for node_id, dot_label in self.cg.nodes.items():
            addr = self.cg.node_addrs.get(node_id)
            name = self.asm.address_to_func_name.get(addr, dot_label)
            id_to_name[node_id] = name

        # Build Adjacency List
        for src_id, dst_id in self.cg.edges:
            src_name = id_to_name.get(src_id, src_id)
            dst_name = id_to_name.get(dst_id, dst_id)
            
            if src_name not in self.call_graph_adj:
                self.call_graph_adj[src_name] = []
            
            if dst_name not in self.call_graph_adj[src_name]:
                self.call_graph_adj[src_name].append(dst_name)
        
        # Ensure all functions appear in keys
        for func_obj in self.asm.functions.values():
            if func_obj.name not in self.call_graph_adj:
                self.call_graph_adj[func_obj.name] = []

        logger.info(f"Call Graph: {len(self.call_graph_adj)} functions")

    def build_cfgs(self):
        """Builds CFGs for each function, correlating DOT nodes with ASM labels"""
        logger.info("Building Control Flow Graphs...")
        
        for func_addr, func_obj in self.asm.functions.items():
            # Locate CFG DOT file: functions/0x<addr>/structuse.dot
            cfg_dir = self.functions_dir / f"0x{func_addr}"
            cfg_path = cfg_dir / "structuse.dot"
            
            if not cfg_path.exists():
                logger.warning(f"CFG DOT not found for {func_obj.name} at {func_addr}")
                continue
            
            parser = CfgDotParser(cfg_path, func_addr)
            if not parser.parse():
                continue
            
            # Map DOT NodeID -> ASM Block Label
            node_id_to_label = {}
            for node_id, info in parser.nodes.items():
                start_addr = info['start']
                label = self.asm.address_to_label.get(start_addr, f"loc_{start_addr}")
                node_id_to_label[node_id] = label
            
            # Build Adjacency List for this function
            func_cfg = {}
            for src_id, dst_id in parser.edges:
                src_label = node_id_to_label.get(src_id, src_id)
                dst_label = node_id_to_label.get(dst_id, dst_id)
                
                if src_label not in func_cfg:
                    func_cfg[src_label] = []
                
                if dst_label not in func_cfg[src_label]:
                    func_cfg[src_label].append(dst_label)
            
            # Ensure all blocks are keys
            for label in node_id_to_label.values():
                if label not in func_cfg:
                    func_cfg[label] = []
            
            self.cfg_adj[func_obj.name] = func_cfg
        
        logger.info(f"CFGs: {len(self.cfg_adj)} functions")

    def to_json(self) -> str:
        output = {
            "metadata": {
                "total_functions": len(self.asm.functions),
                "call_graph_edges": sum(len(v) for v in self.call_graph_adj.values()),
                "cfg_functions": len(self.cfg_adj)
            },
            "call_graph": self.call_graph_adj,
            "control_flow_graphs": self.cfg_adj
        }
        return json.dumps(output, indent=2, sort_keys=False)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    # Parse command line arguments
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('graph_output.json')
    
    logger.info(f"Base directory: {base_dir.resolve()}")
    logger.info(f"Output file: {output_file.resolve()}")
    
    # Define paths based on actual hierarchy
    asm_path = base_dir / 'nasm' / 'structuse.asm'
    cg_path = base_dir / 'structuse.callgraph.dot'
    functions_dir = base_dir / 'functions'
    
    # Validate paths
    required_paths = [
        ('Assembly', asm_path),
        ('Call Graph', cg_path),
        ('Functions Directory', functions_dir)
    ]
    
    for name, path in required_paths:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)
    
    if not functions_dir.is_dir():
        logger.error(f"Functions directory is not a directory: {functions_dir}")
        sys.exit(1)
    
    # 1. Parse Assembly (Source of Truth for Names)
    asm_parser = AsmParser(asm_path)
    if not asm_parser.parse():
        logger.error("Failed to parse assembly file")
        sys.exit(1)

    # 2. Parse Call Graph
    cg_parser = CallGraphDotParser(cg_path)
    if not cg_parser.parse():
        logger.error("Failed to parse call graph")
        sys.exit(1)

    # 3. Build Unified Graphs
    builder = GraphBuilder(asm_parser, cg_parser, functions_dir)
    builder.build_call_graph()
    builder.build_cfgs()

    # 4. Output JSON
    json_output = builder.to_json()
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        logger.info(f"Successfully wrote graph data to {output_file}")
        logger.info(f"Output size: {len(json_output)} bytes")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
