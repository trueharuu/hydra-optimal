const PIECE_CHARS = "IJLOSTZ";
const PIECE_SIZE = 4;
const PC_HEIGHT = 4;
const FIELD_HEIGHT = PIECE_SIZE + PC_HEIGHT;
const FIELD_WIDTH = 10;
const PIECE_SHAPES = 7;

const MINO_TABLE = [
    [
        [[0, -1], [0, 0], [0, 1], [0, 2]],
        [[-1, 1], [0, 1], [1, 1], [2, 1]],
        [[1, 2], [1, 1], [1, 0], [1, -1]],
        [[2, 0], [1, 0], [0, 0], [-1, 0]]
    ],
    [
        [[-1, -1], [0, -1], [0, 0], [0, 1]],
        [[-1, 1], [-1, 0], [0, 0], [1, 0]],
        [[1, 1], [0, 1], [0, 0], [0, -1]],
        [[1, -1], [1, 0], [0, 0], [-1, 0]]
    ],
    [
        [[0, -1], [0, 0], [0, 1], [-1, 1]],
        [[-1, 0], [0, 0], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [0, -1], [1, -1]],
        [[1, 0], [0, 0], [-1, 0], [-1, -1]]
    ],
    [
        [[0, 0], [-1, 0], [-1, 1], [0, 1]],
        [[0, 0], [-1, 0], [-1, 1], [0, 1]],
        [[0, 0], [-1, 0], [-1, 1], [0, 1]],
        [[0, 0], [-1, 0], [-1, 1], [0, 1]]
    ],
    [
        [[0, -1], [0, 0], [-1, 0], [-1, 1]],
        [[-1, 0], [0, 0], [0, 1], [1, 1]],
        [[0, 1], [0, 0], [1, 0], [1, -1]],
        [[1, 0], [0, 0], [0, -1], [-1, -1]]
    ],
    [
        [[0, -1], [0, 0], [-1, 0], [0, 1]],
        [[-1, 0], [0, 0], [0, 1], [1, 0]],
        [[0, 1], [0, 0], [1, 0], [0, -1]],
        [[1, 0], [0, 0], [0, -1], [-1, 0]]
    ],
    [
        [[-1, -1], [-1, 0], [0, 0], [0, 1]],
        [[-1, 1], [0, 1], [0, 0], [1, 0]],
        [[1, 1], [1, 0], [0, 0], [0, -1]],
        [[1, -1], [0, -1], [0, 0], [-1, 0]]
    ]
];

const COLOR_TABLE = [
    "#00ffff",
    "#3040ff",
    "#ffa500",
    "#ffff00",
    "#00ee00",
    "#ff00ff",
    "#ff0000",
    "#d3d3d3",
    "#000000"
];

const GARBAGE = 7;
const EMPTY = 8;

function Piece(shape, rot, row, col) {
    this.shape = shape;
    this.rot = rot;
    this.row = row;
    this.col = col;

    this.get_mino = function (mino) {
        return [
            row + MINO_TABLE[this.shape][this.rot][mino][0],
            col + MINO_TABLE[this.shape][this.rot][mino][1]
        ];
    };
}

function Field(fhash = 0, fill = GARBAGE) {
    this.grid = Array(FIELD_HEIGHT).fill().map(
        () => Array(FIELD_WIDTH).fill(EMPTY)
    );
    for (let r = FIELD_HEIGHT - 1; r >= 0; r--) {
        for (let c = FIELD_WIDTH - 1; c >= 0; c--) {
            this.grid[r][c] = ((fhash % 2) ? fill : EMPTY);
            fhash = Math.floor(fhash / 2);
        }
    }

    this.lines = 0;

    this.can_place = function (p) {
        for (let mino = 0; mino < PIECE_SIZE; mino++) {
            let pos = p.get_mino(mino);
            if (pos[0] < 0 || pos[0] >= FIELD_HEIGHT || pos[1] < 0 || pos[1] >= FIELD_WIDTH) {
                return false;
            }
            if (this.grid[pos[0]][pos[1]] != EMPTY) {
                return false;
            }
        }
        return true;
    }

    this.place = function (p) {
        for (let mino = 0; mino < PIECE_SIZE; mino++) {
            let pos = p.get_mino(mino);
            this.grid[pos[0]][pos[1]] = p.shape;
        }
    };

    this.clear_lines = function () {
        let num_cleared = 0;
        let flag;
        for (let r = FIELD_HEIGHT - 1; r >= (PIECE_SIZE - num_cleared); r--) {
            if (r >= PIECE_SIZE) {
                for (let c = 0; c < FIELD_WIDTH; c++) {
                    flag = (this.grid[r][c] == EMPTY);
                    if (flag) {
                        break;
                    }
                }
                if (!flag) {
                    num_cleared++;
                }
                else if (num_cleared > 0) {
                    for (let c = 0; c < FIELD_WIDTH; c++) {
                        this.grid[r + num_cleared][c] = this.grid[r][c];
                    }
                }
            }
            else if (r >= 0) {
                for (let c = 0; c < FIELD_WIDTH; c++) {
                    this.grid[r + num_cleared][c] = this.grid[r][c];
                }
            }
            else {
                for (let c = 0; c < FIELD_WIDTH; c++) {
                    this.grid[r + num_cleared][c] = EMPTY;
                }
            }
        }
        for (let r = 0; r < FIELD_HEIGHT; r++) {
            for (let c = 0; c < FIELD_WIDTH; c++) {
                if (this.grid[r][c] != EMPTY) {
                    this.grid[r][c] = GARBAGE;
                }
            }
        }
        this.lines += num_cleared;
    };

    this.disp = function (onclick = "") {
        return `<table class="field" cellspacing="0">` +
            this.grid.slice(PIECE_SIZE).map(
                row => `<tr>` + row.map(
                    sq => `<td style="background:${COLOR_TABLE[sq]}"` +
                        ` onclick="${onclick}"></td>`
                ).join("") + `</tr>`
            ).join("") +
            `</table>`;
    };

    this.hash = function () {
        let h = 0;
        for (let r = PIECE_SIZE + this.lines; r < FIELD_HEIGHT; r++) {
            for (let c = 0; c < FIELD_WIDTH; c++) {
                h *= 2;
                h += (this.grid[r][c] != EMPTY);
            }
        }
        h *= Math.pow(2, FIELD_WIDTH * this.lines);
        h += Math.pow(2, FIELD_WIDTH * this.lines) - 1;
        return h;
    }
}

function interpolate(f1, f2, shape) {
    for (let r = 8; r > 2; r--) {
        for (let c = -1; c < 11; c++) {
            for (let t = 0; t < 4; t++) {
                let p = new Piece(shape, t, r, c);
                if (f1.can_place(p)) {
                    let f3 = new Field(f1.hash());
                    f3.clear_lines();
                    f3.place(p);
                    f3.clear_lines();
                    if (f2.hash() == f3.hash()) {
                        return p;
                    }
                }
            }
        }
    }
}

// Reveal-piece glyphs used as the clickable buttons for choosing a child branch.
const PIECE_FIELDS = [
    new Field(535296000, 0),
    new Field(206360015100, 1),
    new Field(12897743100, 2),
    new Field(128974971000, 3),
    new Field(64487670000, 4),
    new Field(51590197500, 5),
    new Field(257949757500, 6)
];

let data;
let init_hash;
let node_pool = [];

function shape_index(ch) {
    return PIECE_CHARS.indexOf(ch);
}

// Rebuild the display field for `to_hash` with the last-placed piece (`piece_char`)
// colored in, interpolated from `from_hash`. The tree hashes are graph hashes (the field
// after the placement and any line clears), so the target is compared without re-clearing.
function placed_field(from_hash, piece_char, to_hash) {
    let f = new Field(from_hash);
    f.clear_lines();
    let f2 = new Field(to_hash);
    let p = interpolate(f, f2, shape_index(piece_char));
    if (p === undefined) {
        return f2;
    }
    f.place(p);
    return f;
}

function leaf_value(obj) {
    if (obj === null || obj === undefined) {
        return "...";
    }
    if (typeof obj === "number") {
        return obj.toFixed(12).replace(/\.?0+$/, "");
    }
    return String(obj.value);
}

// A compact view of the next placement in `obj`'s subtree.
function preview(prev_hash, obj) {
    if (obj === null || obj === undefined) {
        return "?";
    }
    if (typeof obj === "number") {
        return `<p>${leaf_value(obj)}</p>`;
    }
    if (obj.capped === true) {
        return `<p>capped</p>`;
    }
    if (obj.steps !== undefined) {
        if (obj.steps.length === 0) {
            return (new Field(prev_hash)).disp();
        }
        return placed_field(prev_hash, obj.steps[0].piece, obj.steps[0].hash).disp();
    }
    return placed_field(prev_hash, obj.piece, obj.hash).disp();
}

function disp_options(prev_hash, obj) {
    if (data === undefined) {
        init_hash = tree_data.init_hash;
        data = tree_data.root;
    }
    if (obj === undefined) {
        obj = data;
    }
    if (prev_hash === undefined) {
        prev_hash = init_hash;
    }
    node_pool = [];
    document.getElementById("results").innerHTML = render_tree(prev_hash, obj);
}

function render_tree(prev_hash, obj) {
    if (obj === null || obj === undefined) {
        return "How Did We Get Here?";
    }
    if (typeof obj === "number") {
        return `${(new Field(prev_hash)).disp()}<p>${leaf_value(obj)}</p>`;
    }
    if (obj.capped === true) {
        return `${(new Field(prev_hash)).disp()}<p>capped ${leaf_value(obj)}</p>`;
    }
    if (obj.steps !== undefined) {
        let r = prev_hash;
        let fields = [];
        for (let x of obj.steps) {
            let f = placed_field(r, x.piece, x.hash);
            fields.push(f.disp());
            r = x.hash;
        }
        let f = new Field(prev_hash);
        f.clear_lines();
        return `${f.disp()}<p>${obj.value}</p><br>` +
            `<div class="grid">${fields.join("")}</div>`;
    }

    let f = placed_field(prev_hash, obj.piece, obj.hash);
    let fields = [];
    for (let shape = 0; shape < PIECE_SHAPES; shape++) {
        let ch = PIECE_CHARS[shape];
        let child = (obj.children || {})[ch];
        if (child !== undefined) {
            node_pool.push(child);
            let id = node_pool.length - 1;
            fields.push(
                `<div>` +
                PIECE_FIELDS[shape].disp(`disp_options(${obj.hash}, node_pool[${id}])`) +
                `<br>` +
                preview(obj.hash, child) +
                `<p>${leaf_value(child)}</div>`
            );
        }
        else {
            fields.push(
                `<div>` +
                PIECE_FIELDS[shape].disp() +
                `<p> </p></div>`
            );
        }
    }
    return `${f.disp()}<p>${obj.value}</p><br>` +
        `<div class="grid">${fields.join("")}</div>`;
}