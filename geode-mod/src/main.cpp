// Starmeter Live: tells the Starmeter app which Geometry Dash level you have open, and shows its rating on the page.
//
// It writes files into this mod's own save folder:
//   current.json   which level you are looking at (ID, name, creator, stars, ...)
//   level.txt      that level's data, only for levels you are editing yourself
// then asks the Starmeter app on this same PC (http://127.0.0.1:<port>/api/live/rate, never the internet)
// to rate that level, and draws the answer on the level page.

#include <Geode/Geode.hpp>
#include <Geode/loader/Dirs.hpp>
#include <Geode/utils/file.hpp>
#include <Geode/utils/string.hpp>
#include <Geode/utils/web.hpp>
#include <Geode/modify/EditLevelLayer.hpp>
#include <Geode/modify/LevelInfoLayer.hpp>
#include <Geode/modify/LevelSelectLayer.hpp>

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <thread>

using namespace geode::prelude;

namespace {
    int s_seq = 0;      // goes up on every announcement, so Starmeter notices a level opened twice in a row

    struct Shown { std::string value, err; };
    std::map<std::string, Shown> s_rated;   // finished ratings of online/official levels, so going back to a page is instant

    // Where and how the rating is drawn on a page: one row growing right from `at` (the official level card)
    // or one row centred on `at` (the level page and the editor page).
    enum class Look { Left, Center };
    struct Spot {
        CCPoint at;
        Look look = Look::Left;
        float size = 1.f;       // 1 = full size; smaller where the free space is narrow
    };

    CCLabelBMFont* text(std::string const& s, char const* font, float scale, GLubyte opacity = 255) {
        auto l = CCLabelBMFont::create(s.c_str(), font);
        l->setScale(scale);
        l->setOpacity(opacity);
        return l;
    }

    // `kids` in a line, centred in a box `w` wide (or from its left edge for Look::Left); the node's origin is the box's anchor.
    CCNode* line(float w, Look look, float gap, std::initializer_list<CCNode*> kids) {
        auto align = look == Look::Left ? AxisAlignment::Start : AxisAlignment::Center;
        auto n = CCNode::create();
        n->setContentSize({w, 20.f});
        n->setAnchorPoint({look == Look::Left ? 0.f : .5f, .5f});
        n->setLayout(RowLayout::create()->setAxisAlignment(align)->setAutoScale(false)->setGap(gap));
        for (auto k : kids) n->addChild(k);
        n->updateLayout();
        return n;
    }

    // "Starmeter 6.84 * +/-1.3" on a level page. Only a caption means "still working / not available".
    void showBadge(CCNode* parent, Spot spot, std::string const& caption, std::string const& value = "", std::string const& err = "") {
        if (auto old = parent->getChildByID("starmeter-badge")) old->removeFromParent();
        auto badge = CCNode::create();
        badge->setID("starmeter-badge");
        badge->setPosition(spot.at);
        badge->setZOrder(50);
        float k = spot.size;
        auto cap = text(caption, "goldFont.fnt", 0.45f * k, value.empty() ? 190 : 255);
        if (value.empty()) {
            badge->addChild(line(300.f, spot.look, 5.f * k, {cap}));
        }
        else {
            auto v = text(value, "bigFont.fnt", 0.55f * k);
            auto star = CCSprite::createWithSpriteFrameName("GJ_starsIcon_001.png");
            star->setScale(0.75f * k);
            auto e = text(err, "chatFont.fnt", 0.65f * k, 190);
            badge->addChild(line(300.f, spot.look, 5.f * k, {cap, v, star, e}));
        }
        parent->addChild(badge);
    }

    std::string jsonEscape(std::string_view in) {
        std::string out;
        out.reserve(in.size() + 8);
        for (unsigned char c : in) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                        out += buf;
                    }
                    else {
                        out += static_cast<char>(c);
                    }
            }
        }
        return out;
    }

    // kind: "official" (a main level), "online" (a level from the servers) or "local" (one of your own levels).
    // withData: also save the level data, so Starmeter can rate levels that are not online.
    // Returns this announcement's sequence number (0 if there was no level).
    int announce(GJGameLevel* level, std::string_view kind, bool withData) {
        if (!level) return 0;
        auto dir = Mod::get()->getSaveDir();

        bool hasData = false;
        if (withData) {
            std::string data = level->m_levelString;
            if (!data.empty()) {
                auto res = utils::file::writeStringSafe(dir / "level.txt", data);
                if (res.isErr()) log::warn("Could not write level.txt: {}", res.unwrapErr());
                else hasData = true;
            }
        }

        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::string json = "{\"v\":1";
        json += ",\"seq\":" + std::to_string(++s_seq);
        json += ",\"ts\":" + std::to_string(static_cast<long long>(now));
        json += ",\"kind\":\"" + std::string(kind) + "\"";
        json += ",\"id\":" + std::to_string(level->m_levelID.value());
        json += ",\"name\":\"" + jsonEscape(std::string(level->m_levelName)) + "\"";
        json += ",\"creator\":\"" + jsonEscape(std::string(level->m_creatorName)) + "\"";
        json += ",\"stars\":" + std::to_string(level->m_stars.value());
        json += std::string(",\"demon\":") + (level->m_demon.value() != 0 ? "true" : "false");
        json += ",\"gameDir\":\"" + jsonEscape(utils::string::pathToString(dirs::getGameDir())) + "\"";
        json += std::string(",\"data\":") + (hasData ? "true" : "false");
        json += "}";

        auto res = utils::file::writeStringSafe(dir / "current.json", json);
        if (res.isErr()) log::warn("Could not write current.json: {}", res.unwrapErr());
        else log::debug("Starmeter Live: {} {} ({})", kind, std::string(level->m_levelName), level->m_levelID.value());
        return s_seq;
    }

    // What the Starmeter app said (filled in on a worker thread).
    struct Answer {
        int code = 0;          // HTTP status; 0 or negative = no HTTP answer at all
        int seq = -1;          // which announcement it rated
        double star = -1, err = 0;
        std::string error;
    };

    Answer ask(std::string const& url) {
        Answer a;
        auto res = web::WebRequest().timeout(std::chrono::seconds(120)).getSync(url);
        a.code = res.code();
        matjson::Value json = res.json().unwrapOr(matjson::Value());
        a.error = json["error"].asString().unwrapOr("");
        if (res.ok()) {
            a.seq = static_cast<int>(json["live"]["seq"].asInt().unwrapOr(-1));
            a.star = json["rating"]["star"].asDouble().unwrapOr(-1.0);
            a.err = json["rating"]["err50"].asDouble().unwrapOr(0.0);
        }
        return a;
    }

    void onRated(CCNode* target, Spot spot, int seq, std::string const& key, bool cacheable, Answer const& a) {
        if (a.code != 200) {
            log::warn("Starmeter rating failed (HTTP {}): {}", a.code, a.error);
            // no HTTP status at all (connection refused is -7): nothing is listening on that port, so the app isn't running
            showBadge(target, spot, a.code <= 0 ? "Starmeter: start the app" : "Starmeter: no rating");
            return;
        }
        if (a.seq != seq) return;   // the app answered for a newer level; that page asks for itself
        if (a.star < 0) {
            showBadge(target, spot, "Starmeter: no rating");
            return;
        }
        Shown s;
        s.value = fmt::format("{:.2f}", a.star);
        s.err = fmt::format("+/-{:.1f}", a.err);
        if (cacheable) s_rated[key] = s;
        showBadge(target, spot, "Starmeter", s.value, s.err);
    }

    // Announce the level, then ask the Starmeter app to rate it and draw the answer on `target` at `spot`.
    // If the page is gone by the time the answer arrives, it is dropped.
    void track(GJGameLevel* level, std::string_view kind, bool withData, CCNode* target, Spot spot) {
        int seq = announce(level, kind, withData);
        if (!seq) return;
        std::string key = std::string(kind) + std::to_string(level->m_levelID.value());
        bool cacheable = kind != "local";
        auto hit = s_rated.find(key);
        if (cacheable && hit != s_rated.end()) {
            showBadge(target, spot, "Starmeter", hit->second.value, hit->second.err);
            return;
        }
        showBadge(target, spot, "Starmeter: rating...");

        std::string url = fmt::format("http://127.0.0.1:{}/api/live/rate", Mod::get()->getSettingValue<int64_t>("port"));
        WeakRef<CCNode> page = target;
        std::thread([url, page, spot, seq, key, cacheable] {
            Answer a = ask(url);
            queueInMainThread([page, spot, seq, key, cacheable, a] {
                if (auto node = page.lock()) onRated(node, spot, seq, key, cacheable, a);
            });
        }).detach();
    }
}

// The level page: tap a level in a list, search result, daily, gauntlet...
class $modify(LiveLevelInfoLayer, LevelInfoLayer) {
    bool init(GJGameLevel* level, bool challenge) {
        if (!LevelInfoLayer::init(level, challenge)) return false;
        if (level) {
            // the empty band between the creator's name and the play button: it is the same on every level page, while the
            // space under the difficulty face is shared with coins and the width there is only the face's own
            auto win = CCDirector::get()->getWinSize();
            Spot spot { ccp(win.width / 2.f, win.height * 0.805f), Look::Center, 0.8f };
            if (level->m_levelType == GJLevelType::Main) track(level, "official", false, this, spot);
            else if (level->m_levelID.value() > 0) track(level, "online", false, this, spot);
            else track(level, "local", true, this, spot);
        }
        return true;
    }
};

// Your own levels (the editor list), including ones that are not uploaded.
class $modify(LiveEditLevelLayer, EditLevelLayer) {
    bool init(GJGameLevel* level) {
        if (!EditLevelLayer::init(level)) return false;
        // centred between the three big buttons and the length / song / verified row
        auto win = CCDirector::get()->getWinSize();
        track(level, "local", true, this, Spot { ccp(win.width / 2.f, win.height * 0.26f), Look::Center });
        return true;
    }
};

// The official levels carousel (Stereo Madness, Back On Track, ...): follow the page you are on.
class $modify(LiveLevelSelectLayer, LevelSelectLayer) {
    struct Fields {
        int lastPage = -1;
    };

    bool init(int page) {
        if (!LevelSelectLayer::init(page)) return false;
        this->schedule(schedule_selector(LiveLevelSelectLayer::watchPage), 0.4f);
        return true;
    }

    void watchPage(float) {
        if (!m_scrollLayer) return;
        int page = m_scrollLayer->m_page;
        if (page == m_fields->lastPage) return;
        auto levelPage = typeinfo_cast<LevelPage*>(m_scrollLayer->getPage(page));
        if (!levelPage || !levelPage->m_level || !levelPage->m_levelDisplay) return;
        m_fields->lastPage = page;
        // top-left corner of the level card, where the card's own star count sits on the right
        auto card = levelPage->m_levelDisplay;
        auto box = card->boundingBox();
        auto at = levelPage->convertToNodeSpace(card->getParent()->convertToWorldSpace({box.getMinX() + 16.f, box.getMaxY() - 17.f}));
        track(levelPage->m_level, "official", false, levelPage, Spot { at, Look::Left });
    }
};
