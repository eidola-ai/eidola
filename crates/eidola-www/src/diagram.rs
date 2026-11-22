//! Mermaid flowcharts, drawn as inline SVG at build time.
//!
//! A doc keeps its diagram as a fenced `mermaid` block, which GitHub
//! renders on its own. The site reads the same source with `merman`, a
//! Rust implementation of Mermaid 12, takes its semantic model and its
//! layout (ELK's layered algorithm, Mermaid 12's default for flowcharts, so
//! the arrangement matches GitHub's), and draws the SVG here, so that:
//!
//! - colour comes from the theme (`circadian.css` custom properties), and a
//!   diagram follows day and night like the page around it. The source
//!   decides structure, line kind (solid, dotted, thick) and stroke widths;
//!   any colour it names is ignored.
//! - every labelled line is a link to the section of the page with the same
//!   name, and the build fails if that section is missing. A label is a
//!   promise that the page explains the line.
//! - a node with a `click <id> "<target>"` statement links to that target,
//!   rewritten like any other link on the page.
//! - the diagram is an image with a name and a description: `accTitle:` and
//!   `accDescr:` are required.
//!
//! Only flowcharts are drawn. Lines are drawn as the layout routes them, at
//! right angles, with their corners rounded.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use merman::model::SemanticDiagramModel;
use merman::model::families::flowchart::{
    FlowEdgeMarker, FlowEdgeStroke, FlowEdgeVisibility, FlowNode, FlowchartModel,
};
use merman::{OperationControl, RenderOutput, RenderRequest, Renderer, SemanticParser, SvgRequest};
use serde::Deserialize;

/// Space around the whole drawing.
const MARGIN: f64 = 2.0;
/// How far the arrowhead's tip reaches past the end of its path: the marker
/// is 8 user units across, centred on the path's end.
const ARROW_REACH: f64 = 4.0;
/// How far each plate of a stacked node sits from the next. Mermaid draws
/// three plates, and the layout's box holds all of them.
const STACK_OFFSET: f64 = 5.0;
/// The radius a line's corners are rounded to, at most.
const CORNER: f64 = 6.0;
/// Where a boundary's title sits: how far in from the left edge, and the
/// height of the band at the top it is centred in when the layout names none.
const TITLE_INSET: f64 = 10.0;
const TITLE_BAND: f64 = 28.0;
/// Room around a line label's text.
const LABEL_PAD: f64 = 4.0;
/// The widest a figure is on the page: the prose column with its padding,
/// `min(648px, 100%)` in `site.css`.
const COLUMN: f64 = 648.0;
/// How far a diagram wider than the column is shrunk to fit it: no further
/// than 10px text from the 14px it was measured at. A diagram wider than
/// that keeps its natural size and scrolls.
const MIN_FIT_SCALE: f64 = 10.0 / 14.0;

/// What a diagram's text refers to outside itself.
pub struct Context<'a> {
    /// Distinguishes this diagram's element ids from other diagrams' on the
    /// same page.
    pub index: usize,
    /// Every heading id on the page.
    pub sections: &'a [String],
    /// Turns a `click` target into the href the page uses.
    pub link: &'a dyn Fn(&str) -> String,
    /// Turns a line's label into the heading id of its section.
    pub slug: &'a dyn Fn(&str) -> String,
}

/// The parts of merman's flowchart layout this drawing reads. Every box is
/// given by its centre.
#[derive(Deserialize)]
struct LayoutDocument {
    layout: Layout,
}

#[derive(Deserialize)]
enum Layout {
    FlowchartV2(FlowLayout),
}

#[derive(Deserialize)]
struct FlowLayout {
    nodes: Vec<LaidNode>,
    edges: Vec<LaidEdge>,
    clusters: Vec<LaidCluster>,
}

#[derive(Deserialize)]
struct LaidNode {
    id: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    is_cluster: bool,
}

#[derive(Deserialize)]
struct LaidEdge {
    id: String,
    points: Vec<Point>,
    label: Option<CentredBox>,
}

#[derive(Deserialize)]
struct LaidCluster {
    id: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    title: String,
    title_label: Option<CentredBox>,
}

#[derive(Deserialize, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Deserialize, Clone, Copy)]
struct CentredBox {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Rect {
    fn centred(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect {
            x: x - w / 2.0,
            y: y - h / 2.0,
            w,
            h,
        }
    }

    fn union(self, o: Rect) -> Rect {
        let x = self.x.min(o.x);
        let y = self.y.min(o.y);
        Rect {
            x,
            y,
            w: (self.x + self.w).max(o.x + o.w) - x,
            h: (self.y + self.h).max(o.y + o.h) - y,
        }
    }
}

/// The scale a diagram `width` wide is drawn at: 1, or just enough to fit
/// the column when that shrinks it no further than [`MIN_FIT_SCALE`].
fn fit_scale(width: f64) -> f64 {
    let fit = COLUMN / width;
    if (MIN_FIT_SCALE..1.0).contains(&fit) {
        fit
    } else {
        1.0
    }
}

/// The last `stroke-width` among CSS declarations like `stroke-width:3px`,
/// in pixels.
fn stroke_width<'a>(declarations: impl IntoIterator<Item = &'a String>) -> Option<f64> {
    declarations
        .into_iter()
        .filter_map(|d| d.split_once(':'))
        .filter(|(k, _)| k.trim() == "stroke-width")
        .filter_map(|(_, v)| v.trim().trim_end_matches("px").trim().parse().ok())
        .last()
}

/// A node's or line's stroke width: its classes' declarations, then its own.
fn declared_width(model: &FlowchartModel, classes: &[String], styles: &[String]) -> Option<f64> {
    let from_classes = classes
        .iter()
        .filter_map(|c| model.class_defs.get(c))
        .flatten();
    stroke_width(from_classes.chain(styles))
}

/// Drop repeated points and points in the middle of a straight run, so
/// every point left is an end or a corner.
fn corners(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(points.len());
    for &p in points {
        if let Some(&last) = out.last()
            && (p.0 - last.0).abs() < 0.01
            && (p.1 - last.1).abs() < 0.01
        {
            continue;
        }
        if out.len() >= 2 {
            let (a, b) = (out[out.len() - 2], out[out.len() - 1]);
            let cross = (b.0 - a.0) * (p.1 - b.1) - (b.1 - a.1) * (p.0 - b.0);
            if cross.abs() < 0.01 {
                out.pop();
            }
        }
        out.push(p);
    }
    out
}

/// A path through `points` with each corner rounded, by at most `CORNER`
/// and never past the middle of either side.
fn rounded_path(points: &[(f64, f64)]) -> String {
    let p = |(x, y): (f64, f64)| format!("{x:.2},{y:.2}");
    let mut d = format!("M{}", p(points[0]));
    for i in 1..points.len() - 1 {
        let (a, b, c) = (points[i - 1], points[i], points[i + 1]);
        let ab = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        let bc = ((c.0 - b.0).powi(2) + (c.1 - b.1).powi(2)).sqrt();
        let r = CORNER.min(ab / 2.0).min(bc / 2.0);
        let before = (b.0 - (b.0 - a.0) / ab * r, b.1 - (b.1 - a.1) / ab * r);
        let after = (b.0 + (c.0 - b.0) / bc * r, b.1 + (c.1 - b.1) / bc * r);
        let _ = write!(d, "L{}Q{} {}", p(before), p(b), p(after));
    }
    let _ = write!(d, "L{}", p(points[points.len() - 1]));
    d
}

/// Pull a path's end back toward the previous point, so an arrowhead
/// centred on it puts its tip where the path ended.
fn trim_end(points: &mut [(f64, f64)], by: f64) {
    let n = points.len();
    if n < 2 {
        return;
    }
    let (ax, ay) = points[n - 2];
    let (bx, by_) = points[n - 1];
    let len = ((bx - ax).powi(2) + (by_ - ay).powi(2)).sqrt();
    if len > by {
        points[n - 1] = (bx - (bx - ax) / len * by, by_ - (by_ - ay) / len * by);
    }
}

/// The node ids `click` statements name. Mermaid ignores a click on a node
/// that does not exist, so a misspelt one would silently link nothing.
fn clicked_ids(source: &str) -> impl Iterator<Item = &str> {
    source
        .lines()
        .filter_map(|l| l.trim().strip_prefix("click "))
        .filter_map(|rest| rest.split_whitespace().next())
}

fn parse(source: &str) -> Result<FlowchartModel, String> {
    let first = source
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("%%"))
        .unwrap_or("");
    if !(first.starts_with("flowchart") || first.starts_with("graph")) {
        return Err(format!(
            "only flowcharts are drawn, and this diagram starts with `{first}`"
        ));
    }
    let parsed = SemanticParser::new()
        .parse_diagram_sync(source)
        .map_err(|e| e.to_string())?;
    match parsed.model {
        SemanticDiagramModel::Flowchart(model) => Ok(model),
        _ => Err("only flowcharts are drawn".into()),
    }
}

fn lay_out(source: &str) -> Result<FlowLayout, String> {
    let output = Renderer::new()
        .render(RenderRequest::layout_json(
            source,
            OperationControl::new(),
            SvgRequest::default(),
        ))
        .map_err(|e| e.to_string())?;
    let RenderOutput::LayoutJson(Some(layout)) = output else {
        return Err("the diagram has no layout".into());
    };
    let document: LayoutDocument = serde_json::from_value(layout.layout().as_value().clone())
        .map_err(|e| format!("unreadable layout: {e}"))?;
    let Layout::FlowchartV2(layout) = document.layout;
    Ok(layout)
}

/// Lay out `source` and draw it as a `<figure>` holding an inline SVG.
pub fn render(source: &str, cx: &Context) -> Result<String, String> {
    let model = parse(source)?;
    let title = model
        .acc_title
        .clone()
        .filter(|t| !t.trim().is_empty())
        .ok_or("a diagram needs an `accTitle:` line naming it")?;
    let description = model
        .acc_descr
        .clone()
        .filter(|d| !d.trim().is_empty())
        .ok_or("a diagram needs an `accDescr:` line describing it")?;
    let nodes: BTreeMap<&str, &FlowNode> = model.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    for id in clicked_ids(source) {
        if !nodes.contains_key(id) {
            return Err(format!("`{id}` is named in a statement but is not a node"));
        }
    }
    let label_of = |id: &str| {
        nodes
            .get(id)
            .and_then(|n| n.label.clone())
            .unwrap_or_else(|| id.to_string())
    };

    let layout = lay_out(source)?;
    let boxes: BTreeMap<&str, Rect> = layout
        .nodes
        .iter()
        .filter(|n| !n.is_cluster)
        .map(|n| (n.id.as_str(), Rect::centred(n.x, n.y, n.width, n.height)))
        .collect();
    let routes: BTreeMap<&str, &LaidEdge> =
        layout.edges.iter().map(|e| (e.id.as_str(), e)).collect();

    let mut extent: Option<Rect> = None;
    let mut grow = |r: Rect| extent = Some(extent.map_or(r, |e| e.union(r)));
    for r in boxes.values() {
        grow(*r);
    }
    // Boundaries, outermost first: a cluster holding another is larger.
    let mut clusters: Vec<(&LaidCluster, Rect)> = layout
        .clusters
        .iter()
        .map(|c| (c, Rect::centred(c.x, c.y, c.width, c.height)))
        .collect();
    clusters.sort_by(|a, b| {
        (b.1.w * b.1.h)
            .total_cmp(&(a.1.w * a.1.h))
            .then_with(|| a.0.id.cmp(&b.0.id))
    });
    for (_, r) in &clusters {
        grow(*r);
    }

    struct Line {
        path: String,
        dash: Option<&'static str>,
        width: f64,
        start_arrow: bool,
        end_arrow: bool,
        label: Option<(String, Rect)>,
        href: Option<String>,
        tooltip: String,
    }
    let mut lines = Vec::new();
    for e in &model.edges {
        if e.visibility == FlowEdgeVisibility::Invisible {
            continue;
        }
        let route = routes
            .get(e.id.as_str())
            .ok_or_else(|| format!("line {} → {} was not routed", e.from, e.to))?;
        let mut points = corners(&route.points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
        if points.len() < 2 {
            return Err(format!("line {} → {} has no path", e.from, e.to));
        }
        for &(x, y) in &points {
            grow(Rect {
                x,
                y,
                w: 0.0,
                h: 0.0,
            });
        }
        let end_arrow = e.end_marker != FlowEdgeMarker::None;
        let start_arrow = e.start_marker != FlowEdgeMarker::None;
        if end_arrow {
            trim_end(&mut points, ARROW_REACH);
        }
        if start_arrow {
            points.reverse();
            trim_end(&mut points, ARROW_REACH);
            points.reverse();
        }
        let dash = (e.stroke_kind == FlowEdgeStroke::Dotted).then_some("2 5");
        let width = declared_width(&model, &e.classes, &e.style).unwrap_or(match e.stroke_kind {
            FlowEdgeStroke::Thick => 3.0,
            _ => 1.0,
        });
        let mut tooltip = format!("{} → {}", label_of(&e.from), label_of(&e.to));
        let mut href = None;
        let label = match (&e.label, &route.label) {
            (Some(text), Some(b)) if !text.trim().is_empty() => {
                let id = (cx.slug)(text);
                if !cx.sections.contains(&id) {
                    return Err(format!(
                        "the line labelled “{text}” has no section: add a heading “{text}” (id `{id}`) that explains it"
                    ));
                }
                href = Some(format!("#{id}"));
                let _ = write!(tooltip, ": {text}");
                let rect = Rect::centred(b.x, b.y, b.width + 2.0 * LABEL_PAD, b.height);
                grow(rect);
                Some((text.clone(), rect))
            }
            _ => None,
        };
        lines.push(Line {
            path: rounded_path(&points),
            dash,
            width,
            start_arrow,
            end_arrow,
            label,
            href,
            tooltip,
        });
    }
    let extent = extent.ok_or("the diagram is empty")?;
    let (ox, oy) = (extent.x - MARGIN, extent.y - MARGIN);
    let (w, h) = (extent.w + 2.0 * MARGIN, extent.h + 2.0 * MARGIN);
    let scale = fit_scale(w);
    let (dw, dh) = (w * scale, h * scale);

    let n = cx.index;
    let mut s = String::new();
    let _ = write!(
        s,
        "<figure class=\"diagram\"><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{ox:.2} {oy:.2} {w:.2} {h:.2}\" width=\"{dw:.0}\" height=\"{dh:.0}\" role=\"img\" aria-labelledby=\"diagram-{n}-title diagram-{n}-desc\">\
<title id=\"diagram-{n}-title\">{}</title><desc id=\"diagram-{n}-desc\">{}</desc>\
<defs><marker id=\"diagram-{n}-arrow\" viewBox=\"0 0 10 10\" refX=\"5\" refY=\"5\" markerWidth=\"8\" markerHeight=\"8\" markerUnits=\"userSpaceOnUse\" orient=\"auto-start-reverse\"><path class=\"arrowhead\" d=\"M0,0L10,5L0,10z\"/></marker></defs>",
        escape(&title),
        escape(&description),
    );

    s.push_str("<g class=\"boundaries\">");
    for (_, r) in &clusters {
        let _ = write!(
            s,
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" rx=\"6\"/>",
            r.x, r.y, r.w, r.h,
        );
    }
    s.push_str("</g>");
    let open = |s: &mut String, href: &Option<String>, tooltip: &str| match href {
        Some(h) => {
            let _ = write!(
                s,
                "<a href=\"{}\"><title>{}</title>",
                escape(h),
                escape(tooltip)
            );
        }
        None => {
            let _ = write!(s, "<g><title>{}</title>", escape(tooltip));
        }
    };
    let close = |s: &mut String, href: &Option<String>| {
        s.push_str(if href.is_some() { "</a>" } else { "</g>" })
    };

    s.push_str("<g class=\"lines\">");
    for l in &lines {
        open(&mut s, &l.href, &l.tooltip);
        let _ = write!(
            s,
            "<path class=\"hit\" d=\"{}\"/><path class=\"line\" d=\"{}\" stroke-width=\"{}\"",
            l.path, l.path, l.width
        );
        if let Some(dash) = l.dash {
            let _ = write!(s, " stroke-dasharray=\"{dash}\"");
        }
        if l.end_arrow {
            let _ = write!(s, " marker-end=\"url(#diagram-{n}-arrow)\"");
        }
        if l.start_arrow {
            let _ = write!(s, " marker-start=\"url(#diagram-{n}-arrow)\"");
        }
        s.push_str("/>");
        close(&mut s, &l.href);
    }
    // A boundary's title sits in its top-left corner, in the band the layout
    // keeps for it, and is drawn over the lines that cross it.
    s.push_str("</g><g class=\"boundary-titles\">");
    for (c, r) in &clusters {
        let ty = c.title_label.map_or(r.y + TITLE_BAND / 2.0, |t| t.y);
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{ty:.2}\">{}</text>",
            r.x + TITLE_INSET,
            escape(&c.title),
        );
    }
    s.push_str("</g><g class=\"labels\">");
    for l in &lines {
        let Some((text, r)) = &l.label else { continue };
        open(&mut s, &l.href, &l.tooltip);
        let _ = write!(
            s,
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" rx=\"3\"/><text x=\"{:.2}\" y=\"{:.2}\">{}</text>",
            r.x,
            r.y,
            r.w,
            r.h,
            r.x + r.w / 2.0,
            r.y + r.h / 2.0,
            escape(text),
        );
        close(&mut s, &l.href);
    }
    s.push_str("</g><g class=\"nodes\">");
    for node in &model.nodes {
        let Some(&r) = boxes.get(node.id.as_str()) else {
            return Err(format!("node `{}` was not laid out", node.id));
        };
        let label = label_of(&node.id);
        let href = node.link.as_deref().map(|t| (cx.link)(t));
        open(&mut s, &href, &label);
        let width = declared_width(&model, &node.classes, &node.styles).unwrap_or(1.0);
        let stroke = format!(" stroke-width=\"{width}\"");
        let shape = node.shape.as_deref().unwrap_or("square");
        // The box a stacked node's front plate fills, and where its label sits.
        let front = if matches!(
            shape,
            "processes" | "procs" | "st-rect" | "stacked-rectangle"
        ) {
            let plate = |k: f64| Rect {
                x: r.x + k * STACK_OFFSET,
                y: r.y + (2.0 - k) * STACK_OFFSET,
                w: r.w - 2.0 * STACK_OFFSET,
                h: r.h - 2.0 * STACK_OFFSET,
            };
            for k in [2.0, 1.0] {
                let p = plate(k);
                let _ = write!(
                    s,
                    "<rect class=\"node\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" rx=\"4\"{stroke}/>",
                    p.x, p.y, p.w, p.h,
                );
            }
            plate(0.0)
        } else {
            r
        };
        match shape {
            "cylinder" | "cyl" | "db" | "database" => {
                let ry = (r.h * 0.12).min(10.0);
                let (x0, x1, y0, y1) = (r.x, r.x + r.w, r.y + ry, r.y + r.h - ry);
                let rx = r.w / 2.0;
                let _ = write!(
                    s,
                    "<path class=\"node\" d=\"M{x0:.2},{y0:.2}A{rx:.2},{ry:.2} 0 0,0 {x1:.2},{y0:.2}A{rx:.2},{ry:.2} 0 0,0 {x0:.2},{y0:.2}L{x0:.2},{y1:.2}A{rx:.2},{ry:.2} 0 0,0 {x1:.2},{y1:.2}L{x1:.2},{y0:.2}\"{stroke}/>"
                );
            }
            _ => {
                let radius = match shape {
                    "round" | "rounded" => 10.0,
                    "stadium" => front.h / 2.0,
                    _ => 4.0,
                };
                let _ = write!(
                    s,
                    "<rect class=\"node\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" rx=\"{radius:.2}\"{stroke}/>",
                    front.x, front.y, front.w, front.h,
                );
            }
        }
        let _ = write!(
            s,
            "<text x=\"{:.2}\" y=\"{:.2}\">{}</text>",
            front.x + front.w / 2.0,
            front.y + front.h / 2.0,
            escape(&label),
        );
        close(&mut s, &href);
    }
    s.push_str("</g></svg></figure>");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slug(s: &str) -> String {
        s.to_lowercase().replace(' ', "-")
    }

    fn draw(source: &str, sections: &[&str]) -> Result<String, String> {
        let sections: Vec<String> = sections.iter().map(|s| s.to_string()).collect();
        render(
            source,
            &Context {
                index: 0,
                sections: &sections,
                link: &|t| format!("/linked/{t}"),
                slug: &slug,
            },
        )
    }

    const SMALL: &str = "flowchart TB\n  accTitle: Two boxes\n  accDescr: A talks to B.\n  subgraph inside [Inside]\n    a[A]\n  end\n  b@{ shape: processes, label: \"B\" }\n  a talk@-. talks .-> b\n  b --> a\n  click a \"a.md\"\n  classDef wide stroke-width:3px\n  class talk wide\n";

    #[test]
    fn draws_nodes_lines_boundaries_and_links() {
        let svg = draw(SMALL, &["talks"]).unwrap();
        assert!(svg.starts_with("<figure class=\"diagram\"><svg"));
        assert!(svg.contains("<title id=\"diagram-0-title\">Two boxes</title>"));
        assert!(svg.contains("<desc id=\"diagram-0-desc\">A talks to B.</desc>"));
        assert!(svg.contains("<g class=\"boundaries\"><rect"));
        assert!(svg.contains(">Inside</text>"));
        // The labelled line links to its section; the unlabelled one does not.
        assert!(svg.contains("<a href=\"#talks\"><title>A → B: talks</title>"));
        assert!(svg.contains("<g><title>B → A</title>"));
        // A line takes its stroke width from its class, by its id.
        let talks = svg.split("<title>A → B: talks</title>").nth(1).unwrap();
        let talks = talks.split("</a>").next().unwrap();
        assert!(
            talks.contains("stroke-width=\"3\" stroke-dasharray=\"2 5\""),
            "{talks}"
        );
        // The click target goes through the page's link rewriting.
        assert!(svg.contains("<a href=\"/linked/a.md\"><title>A</title>"));
        // The stacked node is three plates: three rects before its label.
        let b = svg.split("<title>B</title>").nth(1).unwrap();
        assert_eq!(
            b.split(">B</text>")
                .next()
                .unwrap()
                .matches("<rect")
                .count(),
            3
        );
        // No colour comes from the diagram; the stylesheet supplies it.
        assert!(!svg.contains("fill=\"#") && !svg.contains("stroke=\"#"));
    }

    #[test]
    fn a_diagram_a_little_wider_than_the_column_shrinks_to_fit() {
        assert_eq!(fit_scale(600.0), 1.0);
        assert_eq!(fit_scale(COLUMN), 1.0);
        assert_eq!(fit_scale(700.0) * 700.0, COLUMN);
        assert_eq!(fit_scale(COLUMN / MIN_FIT_SCALE + 1.0), 1.0);
    }

    #[test]
    fn a_labelled_line_without_a_section_fails() {
        let err = draw(SMALL, &[]).unwrap_err();
        assert!(err.contains("“talks” has no section"), "{err}");
    }

    #[test]
    fn a_diagram_needs_a_name_and_a_description() {
        let untitled = SMALL.replace("  accTitle: Two boxes\n", "");
        assert!(
            draw(&untitled, &["talks"])
                .unwrap_err()
                .contains("accTitle")
        );
        let undescribed = SMALL.replace("  accDescr: A talks to B.\n", "");
        assert!(
            draw(&undescribed, &["talks"])
                .unwrap_err()
                .contains("accDescr")
        );
    }

    #[test]
    fn only_flowcharts_are_drawn() {
        let err = draw("sequenceDiagram\n  A->>B: hi\n", &[]).unwrap_err();
        assert!(err.contains("only flowcharts"), "{err}");
    }

    #[test]
    fn a_statement_naming_no_node_fails() {
        let typo = SMALL.replace("click a ", "click z ");
        assert!(draw(&typo, &["talks"]).unwrap_err().contains("`z`"));
    }

    #[test]
    fn drawing_is_deterministic() {
        let first = draw(SMALL, &["talks"]).unwrap();
        for _ in 0..8 {
            assert_eq!(draw(SMALL, &["talks"]).unwrap(), first);
        }
    }
}
