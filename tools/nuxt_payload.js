#!/usr/bin/env node
/**
 * nuxt_payload.js — 从 Nuxt SSR 页面中安全提取 __NUXT__ 载荷
 *
 * 背景与难点
 * ----------
 * CompTox Chemicals Dashboard 的化学清单页是 Nuxt(Vue) 服务端渲染，
 * 页面里的载荷形如：
 *     __NUXT__=(function(a,b,c,...){ ...数据体... }(v1,v2,v3,...))
 * 其中「数据体」用短变量名引用「参数表」里的字面量值。
 *
 * 对 PFASSTRUCT V6（21,028 条记录）而言参数多达 **91,154 个**，
 * 直接 eval 会触发 V8 的硬限制：
 *     SyntaxError: Too many parameters in function definition (only 65534 allowed)
 *
 * 解决思路（三步，全部不用 eval 整个表达式）：
 *   1. 文本切分：取参数名表 params、函数体 body、调用实参表 args；
 *   2. 用 `[args]` 数组字面量求值（数组元素个数无限制），得到「参数名 → 值」映射；
 *   3. 用 `with(Proxy)` 构造动态作用域执行 body —— 标识符按名解析，避免声明 9 万个形参。
 *
 * 用法：
 *   node nuxt_payload.js <html文件> <输出json> [--key chemicals]
 */
const fs = require('fs');
const vm = require('vm');

const QUOTES = new Set(['"', "'", '`']);

/** 从左括号位置出发做配平扫描，返回配对右括号的下标 */
function matchBracket(src, start, open, close) {
  let depth = 0, i = start, inStr = null, esc = false;
  for (; i < src.length; i++) {
    const ch = src[i];
    if (inStr) {
      if (esc) { esc = false; continue; }
      if (ch === '\\') { esc = true; continue; }
      if (ch === inStr) inStr = null;
      continue;
    }
    if (QUOTES.has(ch)) { inStr = ch; continue; }
    if (ch === open) depth++;
    else if (ch === close) { depth--; if (depth === 0) return i; }
  }
  return -1;
}

function parsePayload(html) {
  const marker = '__NUXT__=';
  const mi = html.indexOf(marker);
  if (mi < 0) throw new Error('未找到 __NUXT__ 标记（页面结构可能已变更）');
  let i = mi + marker.length;
  while (i < html.length && /\s/.test(html[i])) i++;
  if (html[i] !== '(') throw new Error('__NUXT__ 载荷未以 ( 起始');
  const outerEnd = matchBracket(html, i, '(', ')');
  if (outerEnd < 0) throw new Error('载荷括号不配平（页面可能被截断）');
  const expr = html.slice(i, outerEnd + 1);
  return expr;
}

/** 解析 (function(p1,p2,...){BODY}(a1,a2,...)) */
function splitExpr(expr) {
  const s = expr.trim();
  if (!s.startsWith('(')) throw new Error('不是预期的包裹形式');
  let i = 1;
  while (/\s/.test(s[i])) i++;
  if (!s.startsWith('function', i)) {
    // 不是函数式载荷 → 直接就当表达式求值
    return { simple: true, expr: s };
  }
  i += 'function'.length;
  while (/\s/.test(s[i])) i++;
  if (s[i] !== '(') throw new Error('function 后未找到参数表');
  const pEnd = matchBracket(s, i, '(', ')');
  const params = s.slice(i + 1, pEnd).split(',').map((x) => x.trim()).filter(Boolean);
  let j = pEnd + 1;
  while (/\s/.test(s[j])) j++;
  if (s[j] !== '{') throw new Error('参数表后未找到函数体');
  const bEnd = matchBracket(s, j, '{', '}');
  const body = s.slice(j, bEnd + 1);
  let k = bEnd + 1;
  while (/\s/.test(s[k])) k++;
  if (s[k] !== '(') throw new Error('函数体后未找到调用实参表');
  const aEnd = matchBracket(s, k, '(', ')');
  const argText = s.slice(k + 1, aEnd);
  return { simple: false, params, body, argText };
}

function evaluateSimple(expr) {
  const sandbox = { undefined, NaN, Infinity };
  return vm.runInNewContext(expr, sandbox, { timeout: 300000 });
}

function evaluateScoped(params, body, argText) {
  // 1) 实参用数组字面量求值（元素个数不受 V8 形参上限约束）
  const sandbox = { undefined, NaN, Infinity };
  const args = vm.runInNewContext('[' + argText + ']', sandbox, { timeout: 300000 });
  if (args.length !== params.length) {
    console.error(`[warn] 形参 ${params.length} 个 / 实参 ${args.length} 个，按公共长度对齐`);
  }
  const n = Math.min(args.length, params.length);
  const table = Object.create(null);
  for (let i = 0; i < n; i++) table[params[i]] = args[i];

  // 2) 用 with + Proxy 提供动态作用域（既解决 9 万形参限制，
  //    又能吸收数据体里对未声明临时变量的赋值，如 `Egp[0]={...}`）
  const store = Object.create(null);
  Object.assign(store, table);
  const proxy = new Proxy(store, {
    has() { return true; },
    get(t, k) {
      if (typeof k === 'symbol') return undefined;
      if (k in t) return t[k];
      if (Object.prototype.hasOwnProperty.call(globalThis, k)) return globalThis[k];
      const o = {};
      Object.defineProperty(t, k, { value: o, writable: true, enumerable: false });
      return o;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  const fn = new Function('__SCOPE__', 'with(__SCOPE__){' + body + '}');
  return fn(proxy);
}

/**
 * 在对象树里找「记录数组」：元素为对象且字段数 ≥ minKeys 的最长数组。
 * key 只是提示（优先看同名属性），真正的判据是结构形态 —— 载荷外层还套着
 * `data:[{chemicals:[...], chunks:[...]}]` 之类的容器，仅靠 key 会命中错误的层级。
 */
function deepFindArray(root, key) {
  const seen = new Set();
  const stack = [root];
  let best = null;
  let bestScore = -1;
  const consider = (arr, bonus) => {
    if (!Array.isArray(arr) || arr.length < 2) return;
    const o = arr[0];
    if (!o || typeof o !== 'object' || Array.isArray(o)) return;
    const nk = Object.keys(o).length;
    if (nk < 6) return;
    const score = arr.length + bonus;
    if (score > bestScore) { bestScore = score; best = arr; }
  };
  while (stack.length) {
    const cur = stack.pop();
    if (!cur || typeof cur !== 'object' || seen.has(cur)) continue;
    seen.add(cur);
    if (Array.isArray(cur)) {
      consider(cur, 0);
      for (const v of cur) stack.push(v);
    } else {
      for (const k of Object.keys(cur)) {
        const v = cur[k];
        if (Array.isArray(v)) consider(v, k === key ? 1e9 : 0);
        stack.push(v);
      }
    }
  }
  return best;
}

function main() {
  const [htmlPath, outPath, ...rest] = process.argv.slice(2);
  let key = 'chemicals';
  const ki = rest.indexOf('--key');
  if (ki >= 0 && rest[ki + 1]) key = rest[ki + 1];
  if (!htmlPath || !outPath) {
    console.error('用法: node nuxt_payload.js <html> <out.json> [--key chemicals]');
    process.exit(2);
  }
  const html = fs.readFileSync(htmlPath, 'utf8');
  const expr = parsePayload(html);
  const parts = splitExpr(expr);

  let payload;
  if (parts.simple) {
    payload = evaluateSimple(parts.expr);
  } else {
    console.error(`[info] 载荷形参 ${parts.params.length} 个 → 采用 with+Proxy 动态作用域`);
    payload = evaluateScoped(parts.params, parts.body, parts.argText);
  }

  const rows = deepFindArray(payload, key);
  if (!rows) {
    fs.writeFileSync(outPath, JSON.stringify({ _no_rows: true, payload }));
    throw new Error(`未在载荷中找到含 "${key}" 的对象数组`);
  }
  fs.writeFileSync(outPath, JSON.stringify(rows));
  const fields = Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).sort();
  console.error(`[ok] 记录数=${rows.length}  字段=${fields.join(',')}`);
}

main();
