//! Shared, backend-free post fixtures for the chat/space render surface.
//!
//! These build `PostNode` trees (the `get_space_tree` render DTO) directly,
//! with no `Core` and no HTTP, so the chat surface can be exercised both by the
//! visual snapshot cases and by the `eidola_space` example. Depends only on
//! public `eidola_app_core` types — keep it that way so the example (which
//! links the library, not the test tree) can `#[path]`-include it.

use eidola_app_core::{DelegationEnd, PostBlock, PostNode, PostParticipant, PostReference};

/// Build a fixture `PostNode` for the branch/generation snapshot cases. The
/// flattener's depth/branch metadata is supplied directly so the render can be
/// exercised without a backend.
#[allow(clippy::too_many_arguments)]
pub fn fixture_post(
    action_id: &str,
    kind: &str,
    label: &str,
    action_type: &str,
    text: &str,
    depth: usize,
    is_branch: bool,
    generation_count: i64,
) -> PostNode {
    PostNode {
        action_id: action_id.into(),
        item_id: format!("item-{action_id}"),
        parent_action_id: None,
        participant: PostParticipant {
            kind: kind.into(),
            label: label.into(),
        },
        action_type: action_type.into(),
        generation: generation_count - 1,
        generation_count,
        is_current: true,
        model: (kind == "agent").then(|| label.to_string()),
        credits_consumed: (kind == "agent").then_some(700),
        relation: (depth > 0 || is_branch).then(|| "reply".to_string()),
        depth,
        is_branch,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(text.into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    }
}

/// Char-offset range of `phrase` within `content` (for a reference edge's
/// quoted span). Robust to multibyte chars before the phrase.
pub fn char_range(content: &str, phrase: &str) -> (i64, i64) {
    let byte = content
        .find(phrase)
        .unwrap_or_else(|| panic!("reference phrase not found: {phrase:?}"));
    let start = content[..byte].chars().count() as i64;
    (start, start + phrase.chars().count() as i64)
}

/// The "kitchen sink" — a realistic, multi-party space exercising every render
/// element we care about, authored in the flattener's pre-order (spine first,
/// then branches) so it matches what `get_space_tree` would produce. Two humans
/// ("You" + "Mara") and an AI ("kimi-k2") debate Thrasymachus's shepherd from
/// *Republic* I and its bearing on modern paternalism, correcting each other on
/// fact and framing, branching a tangent, quoting one another inline, and
/// collaboratively editing a newsletter excerpt as the product. Used as the
/// non-interactive visual reference for iterating on the chat surface.
pub fn kitchen_sink_posts() -> Vec<PostNode> {
    // Substantive bodies kept in variables so reference spans can be computed
    // against the exact text.
    let a2 = "Three threads worth pulling apart.\n\n\
        First, the position. Thrasymachus's opening definition is that justice is \
        *the advantage of the stronger* (338c): in any regime the ruling party makes \
        laws to its own benefit and calls obedience to them \"just.\" This isn't quite \
        the cartoon of *might makes right* — it's a claim about whose interest the word \
        *justice* actually tracks.\n\n\
        Second, the shepherd. When Socrates answers that every craft seeks the good of \
        its object — medicine the patient's health, not the doctor's — Thrasymachus \
        erupts with the shepherd (343b): shepherds fatten sheep for the feast, not for \
        the sheep, and rulers are no different. The image denies that ruling is a craft \
        aimed at the ruled.\n\n\
        Third, paternalism. The modern \"we know what's good for you\" state is almost \
        the photographic negative of Thrasymachus: the paternalist *adopts* the \
        shepherd's good-of-the-flock language to justify coercion. Thrasymachus would \
        call that the oldest trick — the stronger dressing its advantage in the \
        vocabulary of care.\n\n\
        And yes: Socrates was rattled. Thrasymachus is described bursting in \"like a \
        wild beast,\" and Socrates says he was struck with fear.";

    let a4 = "Fair — I conflated the definition with its defense. Corrected: (1) justice \
        is the advantage of the stronger (338c); (2) Socrates' craft-analogy; (3) *then* \
        the shepherd at 343b, as rebuttal.\n\n\
        Let me also sharpen the \"caught off guard\" point, because it's easy to \
        overstate. Socrates was frightened by his manner, not refuted by his argument — \
        the wild-beast entrance (336b) unsettles him, but he recovers and, by 350d, has \
        forced Thrasymachus into the contradiction that makes him *blush*. So \"caught \
        flat-footed\" is true of the rhetoric and false of the dialectic.";

    let excerpt = "**This week: the shepherd's bargain.** In *Republic* I, Thrasymachus is \
        too often flattened into \"might makes right.\" His sharper move is the shepherd \
        (343b): he denies that ruling is a craft aimed at the good of the ruled, the way \
        medicine aims at health. Modern paternalism runs the image in reverse — it keeps \
        the shepherd's tender vocabulary of care while keeping the shepherd's appetite. \
        The lesson isn't cynicism; it's a question to put to any \"for your own good\": \
        whose advantage does the rule actually track? Even Socrates was frightened by his \
        manner before he answered the argument — force of delivery is not force of reason.";

    // Reference spans, computed against the exact source text.
    let (r3s, r3e) = char_range(a2, "opening definition");
    let (r8s, r8e) = char_range(a4, "frightened by his manner");

    let reply_ref = |antecedent: &str, start: i64, end: i64| PostReference {
        antecedent_action_id: antecedent.into(),
        ordinal: 1,
        content_block_id: None,
        range_start: Some(start),
        range_end: Some(end),
        annotation: None,
        delegation_end: None,
        snippet: None,
        antecedent_author_label: "Ada".into(),
        antecedent_author_kind: "agent".into(),
    };

    // (action_id, item_id, kind, label, type, depth, is_branch, parent, gen, refs, content)
    let post = |action_id: &str,
                kind: &str,
                label: &str,
                atype: &str,
                depth: usize,
                is_branch: bool,
                parent: Option<&str>,
                generation: i64,
                refs: Vec<PostReference>,
                content: &str,
                at: i64|
     -> PostNode {
        let mut n = fixture_post(
            action_id, kind, label, atype, content, depth, is_branch, generation,
        );
        n.parent_action_id = parent.map(String::from);
        n.references = refs;
        n.relation = parent.map(|_| "reply".to_string());
        n.created_at = at;
        n
    };

    vec![
        // Spine.
        post(
            "a1",
            "human",
            "user",
            "user_input",
            0,
            false,
            None,
            1,
            vec![],
            "I keep circling back to Thrasymachus and his shepherd in *Republic* I. The \
             claim, as I remember it, is that rulers tend their subjects the way a shepherd \
             fattens sheep — for the master's table, not the flock's good — and that justice \
             is just a name for whatever serves the stronger. How much of that actually maps \
             onto the modern paternalist state, the \"we know what's good for you\" kind? And \
             is the old story true that Socrates was caught flat-footed by him?",
            1,
        ),
        post(
            "a2",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a1"),
            1,
            vec![],
            a2,
            2,
        ),
        post(
            "a3",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a2"),
            1,
            vec![reply_ref("a2", r3s, r3e)],
            "One correction on the order. You called the shepherd part of his *opening \
             definition*, but it isn't — it's his counter-punch. The definition (advantage of \
             the stronger) comes first; the shepherd only appears at 343b, *after* Socrates \
             presses the craft-analogy. The sequence matters: the shepherd is a rebuttal to a \
             specific argument, not a free-standing thesis.",
            3,
        ),
        post(
            "a4",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a3"),
            1,
            vec![],
            a4,
            4,
        ),
        post(
            "a7",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a4"),
            1,
            vec![],
            "Good — so the newsletter through-line writes itself: Thrasymachus isn't \"might \
             makes right,\" he's denying that ruling is a craft that serves the ruled, and \
             modern paternalism is the inversion — it borrows the shepherd's good-of-the-flock \
             framing to license power. Let me draft the excerpt.",
            7,
        ),
        // The collaboratively edited node (the "product"): drafted by You,
        // refined by Mara — its current tip is Mara's (cross-author edit), so it
        // reads as a v2 generation authored by Mara.
        post(
            "a8",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a7"),
            2,
            vec![reply_ref("a4", r8s, r8e)],
            excerpt,
            9,
        ),
        post(
            "a9",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a8"),
            1,
            vec![],
            "One flag before this ships: the excerpt lets \"advantage of the stronger\" and \
             the shepherd stand as a single move. They're two — the definition (338c) and its \
             defense (343b). A reader who knows the text will catch the seam; otherwise it \
             lands.",
            10,
        ),
        // Branch: a later tangent off a2 (so it indents off the spine).
        post(
            "a5",
            "human",
            "user",
            "user_input",
            1,
            true,
            Some("a2"),
            1,
            vec![],
            "A tangent off your first answer: when you say justice is \"the advantage of the \
             stronger,\" is Thrasymachus making a hard descriptive claim about how power \
             behaves — proto-Machiavelli — or a *definitional* one about what the word means? \
             Those come apart fast.",
            5,
        ),
        post(
            "a6",
            "agent",
            "gemma-4-E2B_q4_0-it@local",
            "inference",
            1,
            false,
            Some("a5"),
            1,
            vec![],
            "Definitional — and that's his vulnerability. If it's merely descriptive (\"the \
             strong usually win\"), it's banal and Socrates has no quarrel. By making it a \
             *definition* of justice, Thrasymachus must defend that the ruler qua ruler never \
             errs about his own advantage, and that's exactly the crack Socrates pries open at \
             340c.",
            6,
        ),
    ]
}

/// The cross-space quote scene: a persisted post quoting one passage from
/// **this** conversation and two from conversations this window has never
/// loaded — the state the footnote rail used to render as three rows all
/// reading "another space" (task 68).
///
/// Each row exercises one case of the rail's naming rule:
///
/// 1. an author this window can see, named by the quoted post's own gutter
///    byline (which is not the label the edge carries — the reader must not be
///    shown two names for one person inside one window);
/// 2. an author only the edge can name, by name (`antecedent_author_label`, the
///    *source* space's effective label);
/// 3. a **blank** label whose *kind* still names the author — the human is
///    "You", an unnamed agent is "Eidola" — which is what makes the composing
///    rail and the persisted rail say the same thing (Codex review, PR #292);
/// 4. a valid but very long label, which is bounded to a share of the row
///    rather than squeezing the passage it attributes out of view.
pub fn cross_space_reference_posts() -> Vec<PostNode> {
    let source = "Justice can't be whatever the strong declare, or the word does no work at \
                  all — it just re-describes who won. The interesting version of Thrasymachus \
                  is the one where the ruler is genuinely competent."
        .to_string();
    let quoted_here = "it just re-describes who won";
    let (here_start, here_end) = byte_range(&source, quoted_here);

    let block = "blk-here";
    // The one post this window holds — the reader's own, whose effective label
    // is the generic `user` and whose gutter therefore reads "You". Row 1 is
    // what makes the rail's first source visible: it says what the gutter two
    // inches up says, not what the edge carries.
    let mut here = fixture_post("c1", "human", "user", "user_input", &source, 0, false, 1);
    here.blocks[0].id = block.into();
    here.created_at = 1;

    let mut reply = fixture_post(
        "c2",
        "human",
        "user",
        "user_input",
        "Four passages, and only one of them is from this conversation.\n\n\
         {{ embed 1 }}\n\nSofia put it the other way round over in the seminar thread:\n\n\
         {{ embed 2 }}\n\nThe rest I'm still chewing on.",
        0,
        false,
        1,
    );
    reply.parent_action_id = Some("c1".into());
    reply.relation = Some("reply".into());
    reply.created_at = 2;
    reply.references = vec![
        PostReference {
            antecedent_action_id: "c1".into(),
            ordinal: 1,
            content_block_id: Some(block.into()),
            range_start: Some(here_start),
            range_end: Some(here_end),
            annotation: None,
            delegation_end: None,
            snippet: Some(quoted_here.into()),
            // What this space calls the participant — carried on every edge,
            // and deliberately *not* what the rail shows for a post it holds
            // (the gutter says "You"; two names for one person inside one
            // window would be worse than the attribution this repairs).
            antecedent_author_label: "user".into(),
            antecedent_author_kind: "human".into(),
        },
        PostReference {
            antecedent_action_id: "x-seminar-1".into(),
            ordinal: 2,
            content_block_id: Some("blk-seminar".into()),
            range_start: Some(0),
            range_end: Some(74),
            annotation: None,
            delegation_end: None,
            snippet: Some(
                "competence is exactly what makes the question sharp, not what dissolves it".into(),
            ),
            antecedent_author_label: "Sofia".into(),
            antecedent_author_kind: "agent".into(),
        },
        PostReference {
            antecedent_action_id: "x-anon-1".into(),
            ordinal: 3,
            content_block_id: Some("blk-anon".into()),
            range_start: Some(0),
            range_end: Some(58),
            annotation: None,
            delegation_end: None,
            snippet: Some("a craft that aims past its object is two crafts, badly joined".into()),
            // A space that overrode this participant's label to empty (the
            // schema's "override to empty"). The kind still names them, and
            // must: the human is the reader, and composing this quote said
            // "You".
            antecedent_author_label: String::new(),
            antecedent_author_kind: "human".into(),
        },
        PostReference {
            antecedent_action_id: "x-long-1".into(),
            ordinal: 4,
            content_block_id: Some("blk-long".into()),
            range_start: Some(0),
            range_end: Some(62),
            annotation: None,
            delegation_end: None,
            snippet: Some("the shepherd fattens the flock for a table it will not sit at".into()),
            // A perfectly valid label, and long enough to eat the row: the
            // byline is capped, the passage keeps its share.
            antecedent_author_label:
                "Republic Book I Close-Reading Group (Tuesdays, in the long room upstairs)".into(),
            antecedent_author_kind: "agent".into(),
        },
    ];

    vec![here, reply]
}

/// A delegation's report as its parent conversation holds it: the owning agent
/// posts, quoting each helper's finding out of the room it opened, and every
/// edge carries the ending the driver recorded. Two branches, so the rail shows
/// what a fan-out actually comes back as.
pub fn delegation_report_posts() -> Vec<PostNode> {
    let asked = fixture_post(
        "d1",
        "human",
        "user",
        "user_input",
        "Before we commit to Friday: is the tide going to be a problem?",
        0,
        false,
        1,
    );
    let mut answer = fixture_post(
        "d2",
        "agent",
        "Navigator",
        "inference",
        "I've asked the two of them to look at it properly.",
        0,
        false,
        2,
    );
    answer.parent_action_id = Some("d1".into());

    let mut report = fixture_post(
        "d3",
        "agent",
        "Navigator",
        "inference",
        "Both came back the same way: Friday morning is fine, Friday evening is not.          Surveyor has the tables; Pilot has the approach.",
        0,
        false,
        3,
    );
    report.parent_action_id = Some("d2".into());
    report.references = vec![
        PostReference {
            antecedent_action_id: "s-1".into(),
            ordinal: 1,
            content_block_id: Some("blk-surveyor".into()),
            range_start: Some(0),
            range_end: Some(66),
            annotation: None,
            delegation_end: Some(DelegationEnd::Paused { depth: 2, limit: 2 }),
            snippet: Some("high water is 06:12 and 18:40; the evening one is the spring".into()),
            antecedent_author_label: "Surveyor".into(),
            antecedent_author_kind: "agent".into(),
        },
        PostReference {
            antecedent_action_id: "s-2".into(),
            ordinal: 2,
            content_block_id: Some("blk-pilot".into()),
            range_start: Some(0),
            range_end: Some(58),
            annotation: None,
            delegation_end: Some(DelegationEnd::Paused { depth: 2, limit: 2 }),
            snippet: Some("we would be crossing the bar on the ebb, which I would not".into()),
            antecedent_author_label: "Pilot".into(),
            antecedent_author_kind: "agent".into(),
        },
    ];
    vec![asked, answer, report]
}

/// The quoted-references scene's source-post body — the text every range in
/// [`quoted_reference_posts`] is measured against. A `fn`, not an inline
/// literal, so a caller (the driver's composing scene) computes the same byte
/// offsets rather than hard-coding them past a multi-byte em dash.
pub fn quoted_reference_source() -> String {
    "The shepherd image is doing more work than it looks. Thrasymachus is not saying rulers \
     are cruel; he is saying that ruling, like shepherding, is a craft whose end lies outside \
     its object. The sheep are fattened, tended, protected — and none of that is for the \
     sheep.\n\n\
     What makes it hard to answer is that the care is real. A paternalist state that genuinely \
     improves lives is not thereby refuted; the question Thrasymachus forces is whose advantage \
     the rule tracks when the two come apart."
        .to_string()
}

/// The passage the composing scene quotes ("the care is real"), as a byte
/// range into [`quoted_reference_source`].
///
/// This module is `include!`d by both the visual cases and the driver example;
/// only the former needs the range, so the example's build sees it as dead.
#[allow(dead_code)]
pub fn quoted_reference_selection() -> std::ops::Range<usize> {
    let src = quoted_reference_source();
    let (s, e) = byte_range(&src, "the care is real");
    s as usize..e as usize
}

/// Byte offsets of `phrase` within `content` (the schema's units, unlike the
/// char-based [`char_range`] the older fixtures use).
pub fn byte_range(content: &str, phrase: &str) -> (i64, i64) {
    let start = content
        .find(phrase)
        .unwrap_or_else(|| panic!("reference phrase not found: {phrase:?}"));
    (start as i64, (start + phrase.len()) as i64)
}

/// One incoming reference in [`quoted_reference_posts`] — the shape the scene
/// converts into `eidola_app_core::IncomingReference` when seeding a `Space`.
pub struct QuotedIncoming {
    pub action_id: String,
    pub block_id: String,
    pub range: (i64, i64),
}

/// The quoted-references scene: a source post whose passages other posts have
/// quoted, and the replies that quote them — the fixture behind the wave-2
/// footnote rail, embed blocks, and source highlights.
///
/// Returns `(posts, incoming)` where `incoming` is the reverse index the
/// source post's highlights are painted from (`(quoted action, references)`).
/// Ranges are **byte** offsets into the source block's text, exactly as the
/// schema stores them.
#[allow(clippy::type_complexity)]
pub fn quoted_reference_posts() -> (Vec<PostNode>, Vec<(String, Vec<QuotedIncoming>)>) {
    let source = quoted_reference_source();
    let src = source.as_str();

    let quoted_a = "the care is real";
    let quoted_b = "whose advantage the rule tracks";
    let (a_start, a_end) = byte_range(src, quoted_a);
    let (b_start, b_end) = byte_range(src, quoted_b);
    // The branch quotes a longer span starting at the same point — an overlap
    // with the reply's, so a click on the shared text is ambiguous (the picker).
    let a_long_end = a_end + 30;

    let block = "blk-1";
    let mut source_post = fixture_post("q1", "agent", "kimi-k2", "inference", src, 0, false, 1);
    source_post.blocks[0].id = block.into();
    source_post.created_at = 1;

    // A reply quoting the first passage: the marker stands as its own
    // paragraph (what the editor renders as a quote block) with prose around it.
    let mut reply = fixture_post(
        "q2",
        "human",
        "user",
        "user_input",
        "That's the sentence I keep snagging on:\n\n{{ embed 1 }}\n\nIf the care is real, \
         doesn't the shepherd analogy quietly concede Socrates' point — that the craft does \
         aim at its object after all?",
        0,
        false,
        1,
    );
    reply.parent_action_id = Some("q1".into());
    reply.relation = Some("reply".into());
    reply.created_at = 2;
    reply.references = vec![PostReference {
        antecedent_action_id: "q1".into(),
        ordinal: 1,
        content_block_id: Some(block.into()),
        range_start: Some(a_start),
        range_end: Some(a_end),
        annotation: None,
        delegation_end: None,
        snippet: Some(quoted_a.into()),
        antecedent_author_label: "kimi-k2".into(),
        antecedent_author_kind: "agent".into(),
    }];

    // A branch quoting the overlapping span plus a second passage — two
    // references on one post (the rail's plural case).
    let mut branch = fixture_post(
        "q3",
        "human",
        "Mara",
        "user_input",
        "Two things, separately.\n\n{{ embed 1 }}\n\nand\n\n{{ embed 2 }}\n\nThe first is a \
         concession; the second is the actual test. I'd keep the second and drop the first.",
        1,
        true,
        1,
    );
    branch.parent_action_id = Some("q1".into());
    branch.relation = Some("reply".into());
    branch.created_at = 3;
    branch.references = vec![
        PostReference {
            antecedent_action_id: "q1".into(),
            ordinal: 1,
            content_block_id: Some(block.into()),
            range_start: Some(a_start),
            range_end: Some(a_long_end),
            annotation: None,
            delegation_end: None,
            snippet: src
                .get(a_start as usize..a_long_end as usize)
                .map(String::from),
            antecedent_author_label: "kimi-k2".into(),
            antecedent_author_kind: "agent".into(),
        },
        PostReference {
            antecedent_action_id: "q1".into(),
            ordinal: 2,
            content_block_id: Some(block.into()),
            range_start: Some(b_start),
            range_end: Some(b_end),
            annotation: None,
            delegation_end: None,
            snippet: Some(quoted_b.into()),
            antecedent_author_label: "kimi-k2".into(),
            antecedent_author_kind: "agent".into(),
        },
    ];

    let incoming = vec![(
        "q1".to_string(),
        vec![
            QuotedIncoming {
                action_id: "q2".into(),
                block_id: block.into(),
                range: (a_start, a_end),
            },
            QuotedIncoming {
                action_id: "q3".into(),
                block_id: block.into(),
                range: (a_start, a_long_end),
            },
            QuotedIncoming {
                action_id: "q3".into(),
                block_id: block.into(),
                range: (b_start, b_end),
            },
        ],
    )];

    (vec![source_post, reply, branch], incoming)
}

/// The **trace-visibility** scene (task 34): a conversation whose activity is
/// worth auditing.
///
/// Returns `(posts, traces)` — the rendered tree plus the parallel trace index
/// `AppCore::space_traces` returns. It carries both anchors the disclosure has
/// to handle: an answered turn's rounds hanging under its own reply (including
/// a navigation-tool descent), and **declines** hanging under the post they
/// answered — the gap, where no post was written at all.
///
/// The last post carries three of them — two agents bowing out, one of them
/// twice — which is the case a single aggregated line could not render
/// honestly: it would have to pick one byline for all three.
pub fn trace_posts() -> (Vec<PostNode>, Vec<eidola_app_core::PostTrace>) {
    use eidola_app_core::{PostTrace, TraceEntry};

    let mut ask = fixture_post(
        "t1",
        "human",
        "user",
        "user_input",
        "Which branch actually settled the sampling question? I've lost the thread.",
        0,
        false,
        1,
    );
    ask.created_at = 1;

    let mut answer = fixture_post(
        "t2",
        "agent",
        "Gemma",
        "inference",
        "The one you opened off my second reply — **#h3f2a9c**, \"temperature vs. \
         top-p\". It runs eight posts and ends with you agreeing to pin top-p at \
         0.9 and leave temperature alone.\n\nThe other branch never came back to \
         it; it drifted into evaluation harnesses.",
        0,
        false,
        1,
    );
    answer.parent_action_id = Some("t1".into());
    answer.relation = Some("reply".into());
    answer.created_at = 2;

    let mut follow_up = fixture_post(
        "t3",
        "human",
        "user",
        "user_input",
        "Mara — you were in that branch. Does that match how you remember it?",
        0,
        false,
        1,
    );
    follow_up.parent_action_id = Some("t2".into());
    follow_up.relation = Some("reply".into());
    follow_up.created_at = 3;

    let answered = PostTrace {
        id: "turn-gemma".into(),
        anchor_action_id: "t2".into(),
        participant_label: "Gemma".into(),
        unanswered: false,
        entries: vec![
            TraceEntry::Tool {
                action_id: "tc1".into(),
                request_id: Some("req-1".into()),
                call_id: "call_1".into(),
                name: "list_branches".into(),
                arguments: "{}".into(),
                result: Some(
                    "2 branches at #h91b02e — #h3f2a9c (8 posts, 2 days ago), \
                     #h7c4411 (3 posts, 5 days ago)"
                        .into(),
                ),
            },
            TraceEntry::Tool {
                action_id: "tc2".into(),
                request_id: Some("req-2".into()),
                call_id: "call_2".into(),
                name: "read_thread".into(),
                arguments: "{\"handle\":\"h3f2a9c\",\"limit\":20}".into(),
                result: Some("posts 1–8 of 8 in #h3f2a9c".into()),
            },
            TraceEntry::Tool {
                action_id: "tc3".into(),
                request_id: Some("req-3".into()),
                call_id: "call_3".into(),
                name: "read_post".into(),
                arguments: "{\"handle\":\"h7c4411\"}".into(),
                result: Some(
                    "#h7c4411 · Mara — \"Let's park sampling and talk about the \
                     harness instead.\""
                        .into(),
                ),
            },
        ],
    };

    // The gap: a turn that ran, looked, and wrote nothing.
    let declined = PostTrace {
        id: "turn-mara".into(),
        anchor_action_id: "t3".into(),
        participant_label: "Mara".into(),
        unanswered: true,
        entries: vec![
            TraceEntry::Tool {
                action_id: "tc4".into(),
                request_id: Some("req-4".into()),
                call_id: "call_4".into(),
                name: "read_thread".into(),
                arguments: "{\"handle\":\"h3f2a9c\"}".into(),
                result: Some("posts 1–8 of 8 in #h3f2a9c".into()),
            },
            TraceEntry::Declined {
                action_id: "d1".into(),
                reason: Some("Gemma's summary matches the branch; nothing to add.".into()),
            },
        ],
    };

    // A second participant bows out of the same post — the fan-out case. Each
    // turn is its own line, so neither agent's activity is credited to the
    // other.
    let also_declined = PostTrace {
        id: "turn-ferris".into(),
        anchor_action_id: "t3".into(),
        participant_label: "Ferris".into(),
        unanswered: true,
        entries: vec![
            TraceEntry::Tool {
                action_id: "tc5".into(),
                request_id: Some("req-5".into()),
                call_id: "call_5".into(),
                name: "list_branches".into(),
                arguments: "{}".into(),
                result: Some("2 branches at #h91b02e".into()),
            },
            TraceEntry::Declined {
                action_id: "d2".into(),
                reason: Some("Mara was in that branch, not me.".into()),
            },
        ],
    };

    // ...and Mara, asked again, bows out again. Two turns by one agent under
    // one post: nothing but the turn's own identity tells them apart.
    let declined_again = PostTrace {
        id: "turn-mara-2".into(),
        anchor_action_id: "t3".into(),
        participant_label: "Mara".into(),
        unanswered: true,
        entries: vec![
            TraceEntry::Tool {
                action_id: "tc6".into(),
                request_id: Some("req-6".into()),
                call_id: "call_6".into(),
                name: "read_post".into(),
                arguments: "{\"handle\":\"h3f2a9c\"}".into(),
                result: Some("#h3f2a9c · you — \"pin top-p at 0.9\"".into()),
            },
            TraceEntry::Tool {
                action_id: "tc7".into(),
                request_id: Some("req-7".into()),
                call_id: "call_7".into(),
                name: "decline".into(),
                arguments: "{\"reason\":\"Still nothing to add.\"}".into(),
                result: Some("Declined. This turn ends without a reply.".into()),
            },
            TraceEntry::Declined {
                action_id: "d3".into(),
                reason: Some("Still nothing to add.".into()),
            },
        ],
    };

    (
        vec![ask, answer, follow_up],
        vec![answered, declined, also_declined, declined_again],
    )
}

/// The Library index behind task 37's destination picker: the conversation
/// being read plus two others to quote into.
#[allow(dead_code)]
pub fn destination_spaces() -> Vec<eidola_app_core::SpaceInfo> {
    let space = |id: &str, title: &str, ts: i64| eidola_app_core::SpaceInfo {
        id: id.into(),
        title: Some(title.into()),
        snippet: None,
        created_at: ts,
        last_activity_at: ts,
        message_count: 6,
        archived_at: None,
    };
    vec![
        space("demo", "Thrasymachus and the shepherd", 3),
        space("tides", "Tides and the moon", 2),
        space("reading", "What to read next", 1),
    ]
}

/// The candidates behind task 37's grant picker: one already-shared agent, and
/// one that works in a single conversation and would have to be shared to join
/// this one.
#[allow(dead_code)]
pub fn grantable_agents() -> Vec<eidola_app_core::GrantableAgent> {
    vec![
        eidola_app_core::GrantableAgent {
            id: "agent-ada".into(),
            label: "Ada".into(),
            shared: true,
            home_space_title: None,
        },
        eidola_app_core::GrantableAgent {
            id: "agent-mara".into(),
            label: "Mara".into(),
            shared: false,
            home_space_title: Some("Tides and the moon".into()),
        },
    ]
}
