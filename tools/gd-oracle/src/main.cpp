// Starmeter Oracle: the real game as a reference for gdsim.py.
//
// On the first main menu it plays every job in <save dir>/jobs/*.job, one after another, and writes
// <save dir>/results/<job>.txt with one line per physics tick:  tick x y yVelocity mode upsideDown onGround mini
// followed by "DEAD <tick> <x> <y> <object id>" or "COMPLETE <tick>". Then it quits without saving the game.
//
// Job file:   level main <id>            (an official level)
//             level file <name>          (a level string in jobs/<name>)
//             press <tick> / release <tick>   (physics steps counted from the start of the attempt)
// It only runs when <save dir>/start.flag exists, and deletes that flag first, so a normal game launch is never touched.

#include <Geode/Geode.hpp>
#include <Geode/modify/MenuLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace geode::prelude;
namespace fs = std::filesystem;

namespace {
    struct Job {
        std::string kind, arg;
        double until = 0;                   // "until <x>": end the job with PASS once the player reaches x alive
        std::vector<std::pair<unsigned, bool>> inputs; // (physics tick, press?) in route order
    };

    std::vector<fs::path> s_queue;
    size_t s_next = 0;
    Job s_job;
    bool s_active = false;
    std::ofstream s_out;
    unsigned s_lastTick = ~0u;
    unsigned s_tick = 0;                // processCommands calls with a new m_currentProgress (about one per physics tick)
    size_t s_in = 0;                    // next input to apply

    fs::path dir() { return Mod::get()->getSaveDir(); }

    bool loadJob(fs::path const& f, Job& j) {
        j = Job{};
        std::ifstream in(f);
        std::string w;
        while (in >> w) {
            if (w == "level") in >> j.kind >> j.arg;
            else if (w == "until") in >> j.until;
            else if (w == "press" || w == "release") {
                unsigned t;
                in >> t;
                j.inputs.push_back({t, w == "press"});
            }
        }
        return !j.kind.empty();
    }

    int modeOf(PlayerObject* p) {
        if (p->m_isShip) return 1;
        if (p->m_isBall) return 2;
        if (p->m_isBird) return 3;
        if (p->m_isDart) return 4;
        if (p->m_isRobot) return 5;
        if (p->m_isSpider) return 6;
        if (p->m_isSwing) return 7;
        return 0;
    }

    void startNext() {
        while (s_next < s_queue.size()) {
            auto f = s_queue[s_next++];
            if (!loadJob(f, s_job)) continue;
            GJGameLevel* lvl = nullptr;
            if (s_job.kind == "main") {
                lvl = GameLevelManager::sharedState()->getMainLevel(std::stoi(s_job.arg), false);
            } else {
                std::ifstream in(dir() / "jobs" / s_job.arg, std::ios::binary);
                std::stringstream ss;
                ss << in.rdbuf();
                lvl = GJGameLevel::create();
                lvl->m_levelString = ss.str();
                lvl->m_levelName = s_job.arg;
                lvl->m_levelType = GJLevelType::Editor;
            }
            if (!lvl) continue;
            fs::create_directories(dir() / "results");
            s_out.open(dir() / "results" / (f.stem().string() + ".txt"), std::ios::trunc);
            s_active = true;
            s_lastTick = ~0u;
            s_tick = 0;
            s_in = 0;
            CCDirector::get()->replaceScene(PlayLayer::scene(lvl, false, false));
            return;
        }
        std::ofstream(dir() / "done.txt") << "done";
        game::exit(false);
    }

    // Between jobs the level is quit the normal way (PlayLayer::onQuit) and the next one starts once the menu is back:
    // swapping scenes straight from one level to the next left other mods' overlays (QOLMod labels) pointing at a dead level.
    class Ticker : public CCObject {
    public:
        int quiet = 0;
        void tick(float) {
            if (PlayLayer::get()) { quiet = 0; return; }
            if (++quiet < 2) return;
            CCDirector::get()->getScheduler()->unscheduleSelector(schedule_selector(Ticker::tick), this);
            startNext();
        }
    };
    Ticker* s_ticker = nullptr;

    void finish(std::string const& line) {
        if (!s_active) return;
        s_active = false;
        s_out << line << '\n';
        s_out.close();
        Loader::get()->queueInMainThread([] {
            if (auto pl = PlayLayer::get()) pl->onQuit();
            if (!s_ticker) s_ticker = new Ticker();
            s_ticker->quiet = 0;
            CCDirector::get()->getScheduler()->scheduleSelector(schedule_selector(Ticker::tick), s_ticker, 0.5f, false);
        });
    }
}

// PlayerObject::update runs once per physics step (processCommands does not: it saw the player move 0 to 2 ticks
// between calls), so inputs and the log are tied to it.
class $modify(OraclePlayer, PlayerObject) {
    void update(float dt) {
        auto pl = PlayLayer::get();
        if (s_active && pl && this == pl->m_player1) {
            s_out << s_tick << ' ' << m_position.x << ' ' << m_position.y << ' ' << m_yVelocity << ' ' << modeOf(this) << ' '
                  << (int)m_isUpsideDown << ' ' << (int)m_isOnGround << ' ' << (m_vehicleSize < 1.f ? 1 : 0) << '\n';
            while (s_in < s_job.inputs.size() && s_tick >= s_job.inputs[s_in].first) {
                // straight to the player: input mods (click between frames) can't reorder it
                if (s_job.inputs[s_in].second) this->pushButton(PlayerButton::Jump);
                else this->releaseButton(PlayerButton::Jump);
                s_in++;
            }
            s_tick++;
            if (s_job.until > 0 && m_position.x >= s_job.until) {      // probe jobs: got this far alive
                PlayerObject::update(dt);
                finish(fmt::format("PASS {} {}", s_tick, m_position.x));
                return;
            }
        }
        PlayerObject::update(dt);
    }
};

class $modify(OraclePlay, PlayLayer) {
    void destroyPlayer(PlayerObject* player, GameObject* obj) {
        if (s_active && obj && obj == m_anticheatSpike) {       // the start-of-attempt anti-cheat check, not a death
            PlayLayer::destroyPlayer(player, obj);
            return;
        }
        PlayLayer::destroyPlayer(player, obj);
        if (s_active && player == m_player1 && player->m_isDead) {
            finish(fmt::format("DEAD {} {} {} {}", m_gameState.m_currentProgress, player->m_position.x, player->m_position.y,
                               obj ? obj->m_objectID : -1));
        }
    }

    void levelComplete() {
        if (s_active) finish(fmt::format("COMPLETE {}", m_gameState.m_currentProgress));
        PlayLayer::levelComplete();
    }
};

class $modify(MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;
        static bool once = false;
        if (!once) {
            once = true;
            std::error_code ec;
            if (!fs::remove(dir() / "start.flag", ec)) return true;
            for (auto& e : fs::directory_iterator(dir() / "jobs", ec))
                if (e.path().extension() == ".job") s_queue.push_back(e.path());
            std::sort(s_queue.begin(), s_queue.end());
            if (!s_queue.empty()) Loader::get()->queueInMainThread([] { startNext(); });
        }
        return true;
    }
};
