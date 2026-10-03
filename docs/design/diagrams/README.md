# Design diagrams

Edit the `.puml` source and commit its generated `.svg` alongside it. Design documents embed the SVG and link to the source. Do not edit SVG files manually.

Render from the repository root with Java and Graphviz installed:

```sh
java -Djava.awt.headless=true -jar /path/to/plantuml.jar -failfast2 -tsvg docs/design/diagrams/op-base.puml
```

The diagram is rendered with PlantUML 1.2026.8 and Graphviz 2.43.0. Inspect the result for overlapping edges and labels, then run the repository's formatting hooks before committing.
