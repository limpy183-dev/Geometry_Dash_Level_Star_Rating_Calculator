// One-shot dump of the game's own object definitions for Starmeter's simulator (gdsim.py).
// On the first main menu it creates every object ID and appends one JSON line per object to
// <save dir>/objdump.jsonl: type, hitbox size and offset, radius, decoration flags.
// Crash-safe: the ID being created is written to next.txt first, so a crash on one ID is skipped on the next launch.

#include <Geode/Geode.hpp>
#include <Geode/modify/MenuLayer.hpp>
#include <fstream>
#include <string>

using namespace geode::prelude;

static constexpr int MAX_ID = 5000;

static void dump() {
    auto dir = Mod::get()->getSaveDir();
    auto next = dir / "next.txt";
    int start = 1;
    if (std::ifstream in(next); in) {
        in >> start;
        if (start > MAX_ID) return;     // finished earlier
        start += 1;                     // the ID in next.txt crashed the game last time: skip it
    }
    std::ofstream out(dir / "objdump.jsonl", std::ios::app);
    for (int id = start; id <= MAX_ID; ++id) {
        { std::ofstream n(next, std::ios::trunc); n << id; }
        auto o = GameObject::createWithKey(id);
        if (!o) continue;
        o->customSetup();                      // what the level loader runs: sets the object type, hitbox size and radius
        o->setPosition({0.f, 0.f});
        o->m_isObjectRectDirty = true;
        o->m_boxOffsetCalculated = false;
        auto r = o->getObjectRect();
        auto bo = o->getBoxOffset();
        CCPoint fo{0.f, 0.f};                  // sprite frame offset: where a trimmed sprite (and its hitbox) sits inside the cell
        if (auto f = o->displayFrame()) fo = f->getOffset();
        out << fmt::format(
            "{{\"id\":{},\"type\":{},\"cls\":{},\"w\":{},\"h\":{},\"rx\":{},\"ry\":{},\"rw\":{},\"rh\":{},\"rad\":{},"
            "\"box\":[{},{}],\"foff\":[{},{}],\"cbox\":[{},{}],\"deco\":{},\"deco2\":{},\"solidcolor\":{},\"invis\":{},\"cw\":{},\"ch\":{}}}\n",
            id, (int)o->m_objectType, (int)o->m_classType, o->m_width, o->m_height, r.origin.x, r.origin.y, r.size.width, r.size.height,
            o->m_objectRadius, bo.x, bo.y, fo.x, fo.y, o->m_customBoxOffset.x, o->m_customBoxOffset.y,
            (int)o->m_isDecoration, (int)o->m_isDecoration2, (int)o->m_isSolidColorBlock, (int)o->m_isInvisibleBlock,
            o->getContentSize().width, o->getContentSize().height);
        out.flush();
    }
    std::ofstream n(next, std::ios::trunc);
    n << MAX_ID + 1;
    log::info("hitbox dump finished");
}

class $modify(MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;
        static bool once = false;
        if (!once) { once = true; dump(); }
        return true;
    }
};
