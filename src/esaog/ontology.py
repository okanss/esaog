"""Ontology construction: core + domain TBoxes, defects and flattening."""
from __future__ import annotations

import hashlib
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS

from .domains import DOMAINS, HUMAN_ONLY_CAPS, cap_index

ROOT = Path(__file__).resolve().parents[2]
ONTO_DIR = ROOT / "ontology"
ESAOG = Namespace("https://w3id.org/esaog/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")
ONTOLOGY_VERSION = "1.0.0"


def dns(domain: str) -> Namespace:
    return Namespace(DOMAINS[domain]["ns"])


def domain_tbox(domain: str) -> Graph:
    """Domain extension: capability and data-type hierarchies with labels/equivalences."""
    d = DOMAINS[domain]
    ns = dns(domain)
    g = Graph()
    g.bind("esaog", ESAOG)
    g.bind(d["prefix"], ns)
    g.bind("skos", SKOS)
    onto = URIRef(d["ns"].rstrip("#"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, OWL.imports, URIRef("https://w3id.org/esaog/core")))
    g.add((onto, OWL.versionInfo, Literal(ONTOLOGY_VERSION)))
    for name, parent, label, alts, gloss in d["caps"]:
        c = ns[name]
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.subClassOf, ns[parent] if parent else ESAOG.Capability))
        g.add((c, RDFS.label, Literal(label, lang="en")))
        g.add((c, SKOS.definition, Literal(gloss, lang="en")))
        for a in alts:
            g.add((c, SKOS.altLabel, Literal(a, lang="en")))
        if name in HUMAN_ONLY_CAPS:
            g.add((c, ESAOG.mustBeRealizedBy, ESAOG.HumanActor))
    for a, b in d["equiv"]:
        g.add((ns[a], OWL.equivalentClass, ns[b]))
    for name, parent in d["types"]:
        t = ns[name]
        g.add((t, RDF.type, OWL.Class))
        g.add((t, RDFS.subClassOf, ns[parent] if parent else ESAOG.DataType))
        g.add((t, RDFS.label, Literal(name, lang="en")))
    return g


def core_graph() -> Graph:
    g = Graph()
    g.parse(ONTO_DIR / "esaog-core.ttl", format="turtle")
    return g


def tbox(domain: str, defects: list | None = None, flatten: bool = False) -> Graph:
    """Full TBox for a domain as seen by a method.

    defects: list of (child, parent) capability/type names whose rdfs:subClassOf or
             owl:equivalentClass axiom is *missing* (ontology incompleteness, SOST
             modelling-defect condition). Applied identically to every semantic method.
    flatten: ablation M5 -- every capability is collapsed into its top-level ancestor
             (hierarchy and equivalences removed; specific classes become aliases of the root).
    """
    g = core_graph() + domain_tbox(domain)
    ns = dns(domain)
    for child, parent in defects or []:
        g.remove((ns[child], RDFS.subClassOf, ns[parent]))
        g.remove((ns[child], OWL.equivalentClass, ns[parent]))
        g.remove((ns[parent], OWL.equivalentClass, ns[child]))
    if flatten:
        parent, _, _, _, _ = cap_index(domain)
        for name in parent:
            root = name
            while parent.get(root):
                root = parent[root]
            c = ns[name]
            for o in list(g.objects(c, RDFS.subClassOf)):
                g.remove((c, RDFS.subClassOf, o))
            for o in list(g.objects(c, OWL.equivalentClass)):
                g.remove((c, OWL.equivalentClass, o))
            g.remove((None, OWL.equivalentClass, c))
            if root == name:
                g.add((c, RDFS.subClassOf, ESAOG.Capability))
            else:  # flattened: specific capability is treated as the coarse root
                g.add((c, OWL.equivalentClass, ns[root]))
    return g


def graph_hash(g: Graph) -> str:
    lines = sorted(f"{s} {p} {o}" for s, p, o in g)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def write_release_files():
    """Serialise the domain TTL modules that are released with the benchmark."""
    for dom in DOMAINS:
        domain_tbox(dom).serialize(ONTO_DIR / f"esaog-{dom}.ttl", format="turtle")


if __name__ == "__main__":
    write_release_files()
